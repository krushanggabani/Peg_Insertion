import pybullet as p
import math

class SimpleAgent:
    """
    Phase 1: move gripper to 0.2 m above peg
    Phase 2: move to grasp configuration (use peg position/orientation/size)
    Phase 3: close gripper and confirm grasp via contact forces
    Phase 4: rotate gripper 90 deg about X and lift by 0.2 m (guarded/slip-aware)
    Phase 5: move near cuboid center at z = 0.2 (guarded/slip-aware)
    Phase 6: admittance-based insertion using peg↔cuboid contact force
    """
    def __init__(self, cfg, gripper_id: int, peg_id: int, cuboid_id: int):
        self.cfg = cfg
        self.gid = gripper_id
        self.pid = peg_id
        self.cid = cuboid_id

        # Phase index
        self.phase = 1
        self.timer = 0
        self.confirm_frames = 0

        # Force thresholds / tolerances
        self.force_contact_min = 3.0     # N per pad to call "touch"
        self.force_confirm_frames = 8    # consecutive frames for confirmed grasp
        self.force_hold_min = 3.0        # N total while moving
        self.z_follow_tol = 0.006        # m: peg must follow gripper in Z
        self.xy_follow_tol = 0.004       # m: peg must follow in XY

        # Motion gains
        self.alpha_fast = 0.15
        self.alpha_slow = 0.01
        self.alpha_slower = 0.005/20

        # Insertion guard
        self.insert_force_max = 12.0     # N safety cap
        self.insert_probe_step = 0.003   # m small downward probe if no contact

        # Jaw presets (meters)
        # You can also read these from cfg if you prefer (cfg.jaw_open_m / cfg.jaw_closed_m)
        self.jaw_open  = 0.010
        self.jaw_close = 0.085

        # Slip history
        self.prev_gz = None
        self.prev_pz = None
        self.prev_gxy = None
        self.prev_pxy = None
        self.lost_grasp_frames = 0
        self.lost_grasp_frames_thresh = 5

        # Admittance (Phase 6)
        self.dt    = getattr(cfg, "time_step", 1.0/240.0)
        self.m_z   = 0.8
        self.b_z   = 30.0
        self.k_z   = 200.0
        self.Fd    = 6.0           # target insertion force
        self.vz_max = 0.02
        self.z_ref = None
        self.v_z   = 0.0
        self.xy_beta  = 0.0015
        self.F_cap_xy = 10.0

        # Cached grasp config (computed in Phase 2)
        self._grasp_opening = self.jaw_open
        self._grasp_height_offset = 0.1125  # how far above peg top to place gripper before close
        self._x_rot_quat_90 = p.getQuaternionFromEuler([math.pi/2, 0, 0])

    # ---------- public ----------
    def reset(self):
        self.phase = 1
        self.timer = 0
        self.confirm_frames = 0
        self.prev_gz = self.prev_pz = None
        self.prev_gxy = self.prev_pxy = None
        self.lost_grasp_frames = 0
        self.z_ref = None
        self.v_z = 0.0

    def act(self, obs, hole_pose_world):
        gp = obs["grip_pos"]; go = obs["grip_orn"]; jaw = obs["grip_jaw_sep"]
        peg_p = obs["peg_pos"]; peg_o = obs["peg_orn"]
        hole_p, hole_o = hole_pose_world

        # Defaults (hold pose)
        target_pos = list(gp)
        target_orn = go
        target_jaw = self.jaw_open

        # Read forces
        f_gp_total, f_left, f_right = self._gripper_peg_forces()
        f_pc_total = self._total_normal_force(self.pid, self.cid)

        # Derived waypoints
        peg_above = (peg_p[0], peg_p[1], peg_p[2] + 0.1 + 0.115)

        # Compute grasp configuration from peg size/orientation (Phase 2 helper)
        bbox_dims = self._peg_bbox_world()  # (dx, dy, dz) in world AABB
        # Identify long / mid / short extents (for a rectangular peg)
        order = sorted([(bbox_dims[0],'x'), (bbox_dims[1],'y'), (bbox_dims[2],'z')], key=lambda t: t[0])
        short_len, short_axis = order[0]
        mid_len,   mid_axis   = order[1]
        long_len,  long_axis  = order[2]

        # Phase-specific logic
        if self.phase == 1:
            # move 0.2m above the peg (open)
            target_pos = self._lerp3(gp, peg_above, self.alpha_fast)
            target_jaw = self.jaw_open
            if self._dist3(gp, peg_above) < 0.01:
                self.phase = 2

        elif self.phase == 2:
            # Move to grasp configuration using peg pose/orientation/size
            # Strategy: approach the peg so that the gripper squeezes along the *short* dimension
            # and center the gripper over the peg's centroid slightly above its top.
            # Opening = short_len + a tiny clearance (2–3 mm)
            clearance = 0.003

        
            grasp_height = (peg_p[0], peg_p[1], peg_p[2] + self._grasp_height_offset)
            target_pos = self._lerp3(gp, grasp_height, self.alpha_slow)
            target_jaw = self.jaw_open
            if abs(gp[2] - grasp_height[2]) < 0.004:
                # Ready to close
                self.confirm_frames = 0
                self.timer = 0
                self.phase = 3

        elif self.phase == 3:
            # Close gripper and confirm grasp via contact forces
            target_pos = gp
            target_jaw = self.jaw_close
            both_ok = (f_left >= self.force_contact_min) and (f_right >= self.force_contact_min)
            self.confirm_frames = self.confirm_frames + 1 if both_ok else 0
            self.timer += 1
            if self.confirm_frames >= self.force_confirm_frames and self.timer > 20:
                # seed follow checks
                self.prev_gz = gp[2]; self.prev_pz = peg_p[2]
                self.prev_gxy = (gp[0], gp[1]); self.prev_pxy = (peg_p[0], peg_p[1])
                self.lost_grasp_frames = 0
                self.phase = 4
                self.timer = 0
            elif self.timer > 360:
                # retry from phase 2
                target_jaw = self._grasp_opening
                self.phase = 2

        elif self.phase == 4:
            # Rotate gripper 90° about X and lift +0.2 (guarded)
            # Desired orientation: current * R_x(90°)
            # target_orn = self._quat_mul(go, self._x_rot_quat_90)
            lift_goal = (gp[0], gp[1], gp[2] + 0.2)

            # Lift slowly; tighten if slip or hold force low
            target_pos = self._lerp3(gp, lift_goal, self.alpha_slower)
            target_jaw = self.jaw_close

            gz, pz = gp[2], peg_p[2]
            if self.prev_gz is not None and self.prev_pz is not None:
                dzg = gz - self.prev_gz
                dzp = pz - self.prev_pz
                if dzg > 0 and (dzg - dzp) > self.z_follow_tol:
                    target_jaw = max(self.jaw_close - 0.0005, 0.0)
            if f_gp_total < self.force_hold_min:
                target_jaw = max(self.jaw_close - 0.0005, 0.0)

            self.prev_gz = gz; self.prev_pz = pz
            self.prev_gxy = (gp[0], gp[1]); self.prev_pxy = (peg_p[0], peg_p[1])

            if self._dist3(gp, lift_goal) < 0.01:
                self.phase = 5

        elif self.phase == 5:
            # Move to cuboid center with z = 0.2 (guarded)
            goal = (hole_p[0], hole_p[1], 0.2)
            # Grasp/slip check while traveling
            if not self._grasp_ok(f_gp_total, f_left, f_right, gp, peg_p):
                self.lost_grasp_frames += 1
                target_pos = (gp[0], gp[1], gp[2] + 0.004)
                target_jaw = max(self.jaw_close - 0.0005, 0.0)
                if self.lost_grasp_frames >= self.lost_grasp_frames_thresh:
                    # fall back: stabilize by going back a phase (re-lift)
                    self.phase = 4
            else:
                self.lost_grasp_frames = 0
                target_pos = self._lerp3(gp, goal, self.alpha_slower)
                target_jaw = self.jaw_close
                if self._dist3(gp, goal) < 0.01:
                    # Initialize admittance state for insertion
                    self.z_ref = gp[2]
                    self.v_z = 0.0
                    self.phase = 6

            self.prev_gz = gp[2]; self.prev_pz = peg_p[2]
            self.prev_gxy = (gp[0], gp[1]); self.prev_pxy = (peg_p[0], peg_p[1])

        elif self.phase ==6:
            # Phase 6: Admittance-based insertion at current XY (near hole center)
            F_meas, n_avg = self._contact_force_and_normal(self.pid, self.cid)
            n_xy = (n_avg[0], n_avg[1])
            n_xy_norm = math.hypot(n_xy[0], n_xy[1]) + 1e-9
            n_xy_unit = (n_xy[0]/n_xy_norm, n_xy[1]/n_xy_norm)

            eF = (self.Fd - F_meas)
            v_next = self.v_z + (self.dt / self.m_z) * (eF - self.b_z * self.v_z - self.k_z * (gp[2] - self.z_ref))
            v_next = max(-self.vz_max, min(self.vz_max, v_next))
            z_next = gp[2] + v_next * self.dt
            if F_meas < 0.5:  # gentle probe if no contact
                z_next = gp[2] - self.insert_probe_step * 0.5

            # lateral compliance to reduce side load
            F_xy = min(F_meas, self.F_cap_xy)
            corr_x = -self.xy_beta * n_xy_unit[0] * (F_xy / self.F_cap_xy)
            corr_y = -self.xy_beta * n_xy_unit[1] * (F_xy / self.F_cap_xy)
            x_des = hole_p[0] + corr_x
            y_des = hole_p[1] + corr_y

            target_pos = self._lerp3(gp, (x_des, y_des, z_next), self.alpha_slower)
            target_jaw = self.jaw_close

            if F_meas > self.insert_force_max:
                # micro back-off
                target_pos = self._lerp3(gp, (hole_p[0], hole_p[1], gp[2] + 0.002), 0.5)
                self.v_z *= 0.5

            self.v_z = v_next
            self.z_ref = 0.999 * self.z_ref + 0.001 * gp[2]

        # ---- Print status every step (phase, positions, forces) ----
        print(f"[SimpleAgent] phase={self._phase_name(self.phase)} "
              f"gp=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
              f"peg=({peg_p[0]:.3f},{peg_p[1]:.3f},{peg_p[2]:.3f}) "
              f"Fgp={f_gp_total:.2f} Fpc={f_pc_total:.2f}")

        return target_pos, target_orn, target_jaw

    # ---------- helpers ----------
    def _phase_name(self, ph: int) -> str:
        return {
            1: "above_peg",
            2: "to_grasp_config",
            3: "close_confirm",
            4: "rotate90X_and_lift",
            5: "to_cuboid_center@z0.2",
            6: "insert_admittance",
        }.get(ph, f"unknown({ph})")

    def _peg_bbox_world(self):
        """Return (dx, dy, dz) from world AABB of the peg."""
        aabb = p.getAABB(self.pid)
        dx = max(0.0, aabb[1][0] - aabb[0][0])
        dy = max(0.0, aabb[1][1] - aabb[0][1])
        dz = max(0.0, aabb[1][2] - aabb[0][2])
        return (dx, dy, dz)

    def _quat_mul(self, q1, q2):
        """Hamilton product q = q1 * q2 (xyzw)."""
        x1,y1,z1,w1 = q1
        x2,y2,z2,w2 = q2
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        )

    def _grasp_ok(self, f_total, f_left, f_right, gp, peg_p) -> bool:
        force_ok = (f_total >= self.force_hold_min) and \
                   (f_left >= 0.6*self.force_contact_min) and \
                   (f_right >= 0.6*self.force_contact_min)

        follow_ok = True
        if self.prev_gz is not None and self.prev_pz is not None:
            dzg = gp[2] - self.prev_gz
            dzp = peg_p[2] - self.prev_pz
            if dzg > 0 and (dzg - dzp) > self.z_follow_tol:
                follow_ok = False
        if self.prev_gxy is not None and self.prev_pxy is not None:
            dgx = gp[0] - self.prev_gxy[0]; dgy = gp[1] - self.prev_gxy[1]
            dpx = peg_p[0] - self.prev_pxy[0]; dpy = peg_p[1] - self.prev_pxy[1]
            if (math.hypot(dgx, dgy) - math.hypot(dpx, dpy)) > self.xy_follow_tol:
                follow_ok = False
        return force_ok and follow_ok

    def _total_normal_force(self, a: int, b: int) -> float:
        return float(sum(cp[9] for cp in p.getContactPoints(bodyA=a, bodyB=b)))

    def _contact_force_and_normal(self, a: int, b: int):
        """(total_normal_force, avg_world_normal_on_B)"""
        cps = p.getContactPoints(bodyA=a, bodyB=b)
        if not cps:
            return 0.0, (0.0, 0.0, 1.0)
        F = 0.0; nx = ny = nz = 0.0
        for cp in cps:
            fn = cp[9]; F += fn
            n = cp[7] if isinstance(cp[7], (list, tuple)) else None
            if n is None: n = (0.0, 0.0, 1.0)
            nx += n[0]*fn; ny += n[1]*fn; nz += n[2]*fn
        return (F, (nx/F, ny/F, nz/F)) if F > 1e-9 else (0.0, (0.0, 0.0, 1.0))

    def _gripper_peg_forces(self):
        """Return (total, left_pad, right_pad) normal forces between gripper and peg."""
        left = right = total = 0.0
        for cp in p.getContactPoints(bodyA=self.gid, bodyB=self.pid):
            fn = cp[9]; total += fn
            if cp[3] % 2 == 0: left = max(left, fn)
            else:              right = max(right, fn)
        return total, left, right

    @staticmethod
    def _dist3(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    @staticmethod
    def _lerp3(a, b, t):
        return (a[0] + t*(b[0]-a[0]),
                a[1] + t*(b[1]-a[1]),
                a[2] + t*(b[2]-a[2]))
