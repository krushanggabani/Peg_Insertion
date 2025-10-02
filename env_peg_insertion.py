# env_peg_insertion.py
import os
import csv
import time
import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import pybullet as p
import pybullet_data
import numpy as np

# Reuse your Robotiq85 class as-is:
# from your_module import Robotiq85
# and your mesh loader
# from your_module import load_mesh

DEFAULT_LOG_HEADERS = [
    # commanded
    "t","cmd_px","cmd_py","cmd_pz","cmd_qx","cmd_qy","cmd_qz","cmd_qw","cmd_jaw_sep",

    # actual gripper (base)
    "grip_px","grip_py","grip_pz","grip_qx","grip_qy","grip_qz","grip_qw","grip_jaw_sep",

    # peg pose
    "peg_px","peg_py","peg_pz","peg_qx","peg_qy","peg_qz","peg_qw",

    # contact forces (sums)
    "f_gripper_peg","f_peg_table","f_peg_cuboid","f_cuboid_table"
]

@dataclass
class EnvConfig:
    assets_root: str
    gui: bool = True
    g: float = -9.81
    time_step: float = 1.0/240.0
    real_time: bool = False                 # Scenario 1: set True for tele-op real-time
    solver_iters: int = 150
    contact_erp: float = 0.2
    contact_slop: float = 0.001
    lateral_friction: float = 1.0
    spinning_friction: float = 0.001
    rolling_friction: float = 0.0
    restitution: float = 0.0

    # hole “goal” box around the cuboid hole (local frame of cuboid base)
    hole_center: Tuple[float,float,float] = (0.0, 0.0, 0.05)  # tweak for your STL
    hole_half_extents: Tuple[float,float,float] = (0.006, 0.006, 0.015)  # ~12x12x30 mm
    max_ori_tilt_deg: float = 7.0

    # jaw separation (meters) to parent joint angle mapping
    jaw_open_m: float = 0.085     # ~ 85 mm
    jaw_closed_m: float = 0.0
    joint_open_rad: float = 0.85
    joint_closed_rad: float = 0.0

    # initial object placement
    peg_start: Tuple[float,float,float] = (0.20, 0.20, 0.05)
    cuboid_start: Tuple[float,float,float] = (0.0, -0.20, 0.05)
    gripper_start: Tuple[float,float,float] = (0.0, 0.50, 0.20)
    gripper_start_rpy: Tuple[float,float,float] = (math.pi, 0.0, 0.0)

    # logging
    log_path: str = "logs/peg_run.csv"


class PegInsertionEnv:
    """
    - Gripper base is kinematic: we set its base pose each step from command.
    - Gripper jaws are actuated via your Robotiq85 gear constraints (position control on parent joint).
    - Peg/cuboid/table are dynamic (except plane and possibly cuboid if you prefer fixed).
    - Logs commanded & actual states + contact forces every step.
    """
    def __init__(self, cfg: EnvConfig,
                 robotiq_cls,
                 load_mesh_fn):
        self.cfg = cfg
        self._robotiq_cls = robotiq_cls
        self._load_mesh_fn = load_mesh_fn

        self.client = p.connect(p.GUI if cfg.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, cfg.g)
        p.setTimeStep(cfg.time_step)
        p.setPhysicsEngineParameter(
            fixedTimeStep=cfg.time_step,
            numSolverIterations=cfg.solver_iters,
            contactERP=cfg.contact_erp,
            collisionFilterMode=1,
            allowedCcdPenetration=0.0,
            contactSlop=cfg.contact_slop
        )
        p.setRealTimeSimulation(1 if cfg.real_time else 0)

        self.plane_id = p.loadURDF("plane.urdf")
        self._apply_dynamics(self.plane_id)

        # Spawn gripper
        self.gripper = self._robotiq_cls(cfg.gripper_start, cfg.gripper_start_rpy)
        self.gripper.load()  # your class parses joints & builds gear constraints

        # Spawn objects
        self.cuboid_id = load_mesh_fn_rel(self._load_mesh_fn, cfg.assets_root, "cuboid.stl",
                                          pos=cfg.cuboid_start, mass=1.0, mesh_scale=0.001, fixed=False)
        self.peg_id    = load_mesh_fn_rel(self._load_mesh_fn, cfg.assets_root, "peg.stl",
                                          pos=cfg.peg_start, mass=0.1, mesh_scale=0.001, fixed=False)
        
        # Put the camera closer and point it to the middle of cuboid and peg
        cub_pos, _ = p.getBasePositionAndOrientation(self.cuboid_id)
        peg_pos, _ = p.getBasePositionAndOrientation(self.peg_id)



        target = [(cub_pos[0] + peg_pos[0]) / 2,
          (cub_pos[1] + peg_pos[1]) / 2,
          (cub_pos[2] + peg_pos[2]) / 2 + 0.05]  # a little above

        p.resetDebugVisualizerCamera(
            cameraDistance=0.35,   # smaller = closer
            cameraYaw=35,          # rotate around horizontally
            cameraPitch=-35,       # tilt downward
            cameraTargetPosition=target
        )

        # Optionally make cuboid heavier / fixed for stability in early testing
        # p.changeDynamics(self.cuboid_id, -1, mass=5.0)  # or set fixed=True above

        # Set material/friction for all links
        for body in [self.gripper.id, self.cuboid_id, self.peg_id]:
            self._apply_dynamics(body)

        # Logging
        os.makedirs(os.path.dirname(cfg.log_path), exist_ok=True)
        self.log_fp = open(cfg.log_path, "w", newline="")
        self.logger = csv.writer(self.log_fp)
        self.logger.writerow(DEFAULT_LOG_HEADERS)

        # Internals
        self.ticks = 0
        self._last_cmd = None  # (pos, orn, jaw_sep_m)

    # ---------- public API ----------
    def reset(self):
        # reset base poses
        p.resetBasePositionAndOrientation(self.gripper.id,
            self.cfg.gripper_start,
            p.getQuaternionFromEuler(self.cfg.gripper_start_rpy))
        p.resetBasePositionAndOrientation(self.cuboid_id, self.cfg.cuboid_start, [0,0,0,1])
        p.resetBasePositionAndOrientation(self.peg_id, self.cfg.peg_start, [0,0,0,1])

        # open gripper
        self.command_jaw_separation(self.cfg.jaw_open_m)

        # small sim settle
        for _ in range(120):
            p.stepSimulation()
        self.ticks = 0

    def close(self):
        self.log_fp.close()
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def command_base_pose(self, pos_xyz, orn_xyzw):
        """Kinematic 'teleport' of gripper base each step (still interacts physically via links)."""
        p.resetBasePositionAndOrientation(self.gripper.id, pos_xyz, orn_xyzw)

    def command_jaw_separation(self, jaw_sep_m: float):
        """Map desired jaw separation (meters) to parent joint angle (radians)."""
        a0, a1 = self.cfg.joint_closed_rad, self.cfg.joint_open_rad
        s0, s1 = self.cfg.jaw_closed_m,   self.cfg.jaw_open_m
        # linear map clamp
        t = (jaw_sep_m - s0) / max(1e-9, (s1 - s0))
        t = max(0.0, min(1.0, t))
        target_angle = a0 + t*(a1 - a0)
        self.gripper.move_gripper(target_angle)

    def step(self,
             cmd_pos_xyz: Tuple[float,float,float],
             cmd_orn_xyzw: Tuple[float,float,float,float],
             cmd_jaw_sep_m: float,
             do_sim: bool = True):
        """Single control step; applies command, steps physics (if do_sim), logs everything."""
        self._last_cmd = (cmd_pos_xyz, cmd_orn_xyzw, cmd_jaw_sep_m)

        self.command_base_pose(cmd_pos_xyz, cmd_orn_xyzw)
        self.command_jaw_separation(cmd_jaw_sep_m)

        if do_sim and not self.cfg.real_time:
            p.stepSimulation()
        if self.cfg.real_time:
            # keep real-time pacing stable (PyBullet handles it when enabled)
            pass

        self._log_once()
        self.ticks += 1

        return self.get_obs()

    # ---------- observation & logging ----------
    def get_obs(self) -> Dict:
        grip_pos, grip_orn = p.getBasePositionAndOrientation(self.gripper.id)
        peg_pos, peg_orn   = p.getBasePositionAndOrientation(self.peg_id)
        # approximate jaw separation from parent angle (inverse of mapping)
        parent_angle = p.getJointState(self.gripper.id, self.gripper.mimic_parent_idx)[0]
        jaw_sep_est = self._angle_to_sep(parent_angle)

        obs = dict(
            grip_pos=grip_pos,
            grip_orn=grip_orn,
            grip_jaw_sep=jaw_sep_est,
            peg_pos=peg_pos,
            peg_orn=peg_orn
        )
        return obs

    def _log_once(self):
        # commanded
        if self._last_cmd is None:
            return
        (cpos, corn, csep) = self._last_cmd

        # actual
        grip_pos, grip_orn = p.getBasePositionAndOrientation(self.gripper.id)
        parent_angle = p.getJointState(self.gripper.id, self.gripper.mimic_parent_idx)[0]
        jaw_sep_est = self._angle_to_sep(parent_angle)

        peg_pos, peg_orn = p.getBasePositionAndOrientation(self.peg_id)

        # contacts (sum normal forces)
        f_gp = self._sum_contact_force(self.gripper.id, self.peg_id)
        f_pt = self._sum_contact_force(self.peg_id, self.plane_id)
        f_pc = self._sum_contact_force(self.peg_id, self.cuboid_id)
        f_ct = self._sum_contact_force(self.cuboid_id, self.plane_id)

        row = [
            self.ticks,
            *cpos, *corn, csep,
            *grip_pos, *grip_orn, jaw_sep_est,
            *peg_pos, *peg_orn,
            f_gp, f_pt, f_pc, f_ct
        ]
        self.logger.writerow(row)

    # ---------- success / termination ----------
    def is_inserted(self, pos_tol=0.004, ori_deg_tol=None) -> bool:
        """Simple geometric check: peg COM near hole center & tilt small."""
        if ori_deg_tol is None:
            ori_deg_tol = self.cfg.max_ori_tilt_deg

        peg_pos, peg_orn = p.getBasePositionAndOrientation(self.peg_id)
        cub_pos, cub_orn = p.getBasePositionAndOrientation(self.cuboid_id)

        # Express peg position in cuboid frame
        peg_in_world = np.array(peg_pos)
        cub_pos = np.array(cub_pos)
        cub_R = np.array(p.getMatrixFromQuaternion(cub_orn)).reshape(3,3)
        peg_local = cub_R.T @ (peg_in_world - cub_pos)

        hc = np.array(self.cfg.hole_center)
        he = np.array(self.cfg.hole_half_extents)
        within = np.all(np.abs(peg_local - hc) <= (he + pos_tol))

        # Orientation: require peg z-axis roughly aligned with cuboid z
        peg_R = np.array(p.getMatrixFromQuaternion(peg_orn)).reshape(3,3)
        # world z for peg: third column
        peg_z_world = peg_R[:,2]
        # world z for cuboid:
        cub_z_world = cub_R[:,2]
        ang = math.degrees(math.acos(np.clip(np.dot(peg_z_world, cub_z_world), -1.0, 1.0)))
        ok_ori = (abs(ang) <= ori_deg_tol)

        # And ensure contacts with cuboid are present (to avoid success while floating)
        in_contact = self._sum_contact_force(self.peg_id, self.cuboid_id) > 1.0

        return bool(within and ok_ori and in_contact)

    # ---------- utilities ----------
    def _apply_dynamics(self, body_id: int):
        num_j = p.getNumJoints(body_id)
        for link in [-1] + list(range(num_j)):
            p.changeDynamics(
                body_id, link,
                lateralFriction=self.cfg.lateral_friction,
                spinningFriction=self.cfg.spinning_friction,
                rollingFriction=self.cfg.rolling_friction,
                restitution=self.cfg.restitution,
                contactStiffness=1e6, contactDamping=1e3 # robust contact
            )

    def _sum_contact_force(self, a: int, b: int) -> float:
        total = 0.0
        for cp in p.getContactPoints(bodyA=a, bodyB=b):
            total += cp[9]  # normalForce
        return float(total)

    def _angle_to_sep(self, angle_rad: float) -> float:
        a0, a1 = self.cfg.joint_closed_rad, self.cfg.joint_open_rad
        s0, s1 = self.cfg.jaw_closed_m,   self.cfg.jaw_open_m
        t = (angle_rad - a0) / max(1e-9, (a1 - a0))
        t = max(0.0, min(1.0, t))
        return float(s0 + t*(s1 - s0))
    

def load_mesh_fn_rel(load_mesh_fn, root, name, **kwargs):
    path = os.path.join(root, name)
    return load_mesh_fn(path, **kwargs)
