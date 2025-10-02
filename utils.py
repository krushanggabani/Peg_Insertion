import pybullet as p
import pybullet_data
from collections import namedtuple
import os, time


ASSETS_ROOT = "/home/krushang/Desktop/Research/OLD/Extra_env/Bullet_Project/assets"


# ---------- Base robot ----------
class RobotBase(object):
    """
    Minimal base class that can also support 'gripper-only' robots by setting arm_num_dofs = 0.
    """

    def __init__(self, pos, ori):
        """
        pos: [x, y, z]
        ori: [roll, pitch, yaw] in radians
        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

        # Defaults for 'gripper-only' robots; subclasses can override
        self.arm_num_dofs = 0
        self.arm_rest_poses = []
        self.eef_id = None

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        # Print joints for debugging
        print("\n[Joints]")
        for j in self.joints:
            print(f"{j.id:2d}  {j.name:30s} type={j.type}  lim=({j.lowerLimit:.3f},{j.upperLimit:.3f})")
        print()

    def step_simulation(self):
        # Should be hooked by env loop if needed; here we just call p.stepSimulation() directly
        p.stepSimulation()

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo',
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # REVOLUTE, PRISMATIC, SPHERICAL, PLANAR, FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                # Disable default motor; we'll enable as needed
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints.append(jointInfo(jointID, jointName, jointType, jointDamping, jointFriction,
                                         jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable))
            
             # --- print all joints nicely ---
        print("\n[Joint Listing]")
        for j in self.joints:
            print(f"ID {j.id:2d} | {j.name:30s} | Type {j.type} | Limits ({j.lowerLimit:.3f},{j.upperLimit:.3f})")

        # Arm support is optional
        if self.arm_num_dofs > 0:
            assert len(self.controllable_joints) >= self.arm_num_dofs, "Not enough controllable joints for the arm."
            self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
            self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
            self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
            self.arm_joint_ranges = [up - lo for lo, up in zip(self.arm_lower_limits, self.arm_upper_limits)]
        else:
            self.arm_controllable_joints = []
            self.arm_lower_limits = []
            self.arm_upper_limits = []
            self.arm_joint_ranges = []

    def __init_robot__(self):
        raise NotImplementedError

    def __post_load__(self):
        pass

    # Arm helpers (won't be used if arm_num_dofs == 0)
    def reset_arm(self):
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)
        for _ in range(10):
            self.step_simulation()

    def get_joint_obs(self):
        positions, velocities = [], []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = None
        if self.eef_id is not None and self.eef_id >= 0:
            ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)


# ---------- Robotiq 2F-85 (gripper-only) ----------
class Robotiq85(RobotBase):
    def __init_robot__(self):
        # Load URDF
        urdf_path = "/home/krushang/Desktop/Research/Roboforce/Peg_Insertion/assets/urdf/robotiq_85.urdf"
        self.id = p.loadURDF(
            urdf_path,
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )


        # Gripper is standalone: no arm DoFs
        self.arm_num_dofs = 0
        self.arm_rest_poses = []

        # Reasonable aperture range (in radians) for the parent revolute DOF
        self.gripper_range = [0.0, 0.85]  # adjust to your URDF's limits if different

    def __post_load__(self):
        # Build name -> index map
        self.name_to_idx = {p.getJointInfo(self.id, j)[1].decode(): j for j in range(p.getNumJoints(self.id))}

        # Choose an EE link for observation (optional): use left inner finger pad if present
        eef_link_name_candidates = [
            "left_inner_finger_pad", "right_inner_finger_pad",
            "left_inner_finger", "right_inner_finger"
        ]
        self.eef_id = None
        for nm in eef_link_name_candidates:
            # link index equals joint index of the joint that leads to that link
            if nm in self.name_to_idx:
                self.eef_id = self.name_to_idx[nm]
                break

        # --- Mimic setup via gear constraints ---
        parent_name = "finger_joint"  # main actuated joint; adjust if your URDF differs
        if parent_name not in self.name_to_idx:
            raise RuntimeError(f"Couldn't find parent joint '{parent_name}'. Available: {list(self.name_to_idx.keys())}")

        self.mimic_parent_idx = self.name_to_idx[parent_name]

        # Map of child joints and their multipliers relative to parent
        MIMIC_CHILDREN = {
            'right_outer_knuckle_joint':  1,
            'left_inner_knuckle_joint':   1,
            'right_inner_knuckle_joint':  1,
            'left_inner_finger_joint':   -1,
            'right_inner_finger_joint':  -1,
        }

        # Disable motors so constraints control them
        for nm, mult in MIMIC_CHILDREN.items():
            if nm not in self.name_to_idx:
                raise RuntimeError(f"Mimic child joint '{nm}' not found in URDF.")
            jidx = self.name_to_idx[nm]
            p.setJointMotorControl2(self.id, jidx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        # Also disable parent motor; we'll command it explicitly
        p.setJointMotorControl2(self.id, self.mimic_parent_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        # Create gear constraints
        self._gear_cids = []
        for nm, mult in MIMIC_CHILDREN.items():
            jidx = self.name_to_idx[nm]
            cid = p.createConstraint(
                parentBodyUniqueId=self.id, parentLinkIndex=self.mimic_parent_idx,
                childBodyUniqueId=self.id,  childLinkIndex=jidx,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            # childAngle = gearRatio * parentAngle
            p.changeConstraint(cid, gearRatio=-mult, maxForce=100, erp=1.0)
            self._gear_cids.append(cid)

    # Public API
    def move_gripper(self, angle_rad):
        """Set the parent joint angle; children follow via gear constraints."""
        lo, hi = self.gripper_range
        angle = max(lo, min(hi, angle_rad))
        p.setJointMotorControl2(
            bodyIndex=self.id,
            jointIndex=self.mimic_parent_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle,
            positionGain=0.6,
            force=80.0
        )

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])



def load_mesh(filename, pos=(0, 0, 0), mass=1.0, mesh_scale=1.0, fixed=False):
    if not os.path.isabs(filename):
        filename = os.path.join(ASSETS_ROOT, filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Mesh not found: {filename}")
    scale_vec = [mesh_scale] * 3
    col = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=scale_vec)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=filename, meshScale=scale_vec, rgbaColor=[0.2, 0.6, 0.9, 1.0])
    body = p.createMultiBody(
        baseMass=(0.0 if fixed else float(mass)),
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(pos),
    )
    return body



