import pybullet as p
import math

class SimpleAgent:
    """
    Force-aware scripted agent with admittance control for insertion:
      0) go above peg (open)
      1) descend (open)
      2) close & confirm grasp via contact force (sustained)
      3) lift slowly while monitoring force/slip
      4) move above hole center slowly (checks grasp/slip)
      5) guarded insertion using admittance on peg↔cuboid force (+ lateral compliance)
    """
    def __init__(self, cfg, gripper_id: int, peg_id: int, cuboid_id: int):
        self.cfg = cfg
        self.gid = gripper_id
        self.pid = peg_id
        self.cid = cuboid_id

        # phases / timers
        self.phase = 0
        self.timer = 0
        self.confirm_frames = 0

        # thresholds (tune for your scene scale)
        self.force_contact_min = 3.0          # N per finger to consider "touch" at close
        self.force_confirm_frames = 8         # consecutive frames to confirm grasp
        self.force_hold_min = 1.0             # N total to consider still holding
        self.lift_follow_tol = 0.006          # m: peg should follow gripper in Z within this tol
        self.xy_follow_tol   = 0.004          # m: peg should follow gripper in XY within this tol

        self.insert_force_max = 12.0          # N: if peg-cuboid force spikes above, pause/adjust (safety)
        self.insert_step      = 0.003         # m per control step downward (used before contact)
        self.travel_alpha      = 0.12         # generic lerp gains
        self.travel_alpha_slow = 0.005

        # jaw presets (meters)
        self.jaw_open  = 0.010
        self.jaw_close = 0.085

        # history for slip
        self.prev_gz = None
        self.prev_pz = None
        self.prev_gxy = None
        self.prev_pxy = None

        # hysteresis for grasp loss during motion (phase 4)
        self.lost_grasp_frames = 0
        self.lost_grasp_frames_thresh = 5

        # ---------------- Admittance control (Phase 5) ----------------
        self.dt = getattr(cfg, "time_step", 1.0/240.0)  # simulation step
        # virtual mechanics (tune conservatively first)
        self.m_z = 0.8      # virtual mass
        self.b_z = 30.0     # virtual damping
        self.k_z = 200.0    # virtual stiffness about z_ref
        self.Fd  = 6.0      # desired insertion force [N]
        self.vz_max = 0.02  # max insertion speed [m/s]
        self.z_ref = None   # hold last "neutral" z for admittance
        self.v_z   = 0.0    # admittance state

        # lateral compliance gain (reduce side load using contact normal)
        self.xy_beta   = 0.0015   # meters per unit normalized lateral force cue
        self.F_cap_xy  = 10.0     # saturate influence

    # ---- lifecycle ----
    def reset(self):
        self.phase = 0
        self.timer = 0
        self.confirm_frames = 0
        self.prev_gz = None
        self.prev_pz = None
        self.prev_gxy = None
        self.prev_pxy = None
        self.lost_grasp_frames = 0
        # reset admittance
        self.z_ref = None
        self.v_z   = 0.0

    # ---- public step ----
    def act(self, obs, hole_pose_world):
        gp = obs["grip_pos"]; go = obs["grip_orn"]; jaw = obs["grip_jaw_sep"]
        peg_p = obs["peg_pos"]; peg_o = obs["peg_orn"]
        hole_p, hole_o = hole_pose_world

        target_pos = list(gp)
        target_orn = go
        target_jaw = self.jaw_open

        # waypoints
        peg_above  = (peg_p[0], peg_p[1], peg_p[2] + 0.2)
        peg_touch  = (peg_p[0], peg_p[1], peg_p[2] + 0.12)
        hole_above = (hole_p[0], hole_p[1], hole_p[2] + 0.20)
        lift_goal  = (peg_p[0], peg_p[1], hole_p[2] + 0.2)

        # contact forces
        f_gp_total, f_left, f_right = self._gripper_peg_forces()   # gripper↔peg
        f_pc_total = self._total_normal_force(self.pid, self.cid)  # peg↔cuboid

        # ----------------- PHASES -----------------
        if self.phase == 0:  # open, go above peg
            target_pos = self._lerp3(gp, peg_above, 0.15)
            target_jaw = self.jaw_open
            if self._dist3(gp, peg_above) < 0.01:
                self.phase = 1

        elif self.phase == 1:  # descend to gentle touch height
            target_pos = self._lerp3(gp, peg_touch, 0.15)
            target_jaw = self.jaw_open
            if abs(gp[2] - peg_touch[2]) < 0.004:
                self.phase = 2
                self.confirm_frames = 0
                self.timer = 0

        elif self.phase == 2:  # close & confirm grasp using force on both fingers
            target_pos = gp
            target_jaw = self.jaw_close
            both_ok = (f_left >= self.force_contact_min) and (f_right >= self.force_contact_min)
            self.confirm_frames = self.confirm_frames + 1 if both_ok else 0
            self.timer += 1
            if self.confirm_frames >= self.force_confirm_frames and self.timer > 20:
                self.prev_gz = gp[2]; self.prev_pz = peg_p[2]
                self.prev_gxy = (gp[0], gp[1]); self.prev_pxy = (peg_p[0], peg_p[1])
                self.lost_grasp_frames = 0
                self.phase = 3
                self.timer = 0
            elif self.timer > 360:
                target_pos = (gp[0], gp[1], gp[2] + 0.03)
                target_jaw = self.jaw_open
                self.phase = 0

        elif self.phase == 3:  # lift slowly while monitoring slip (force & follow)
            target_pos = self._lerp3(gp, lift_goal, self.travel_alpha_slow/10)
            target_jaw = self.jaw_close

            gz, pz = gp[2], peg_p[2]
            if self.prev_gz is not None and self.prev_pz is not None:
                dzg = gz - self.prev_gz
                dzp = pz - self.prev_pz
                if dzg > 0 and (dzg - dzp) > self.lift_follow_tol:
                    target_jaw = max(self.jaw_close - 0.0005, 0.0)
            if f_gp_total < self.force_hold_min:
                target_jaw = max(self.jaw_close - 0.0005, 0.0)

            self.prev_gz = gz; self.prev_pz = pz
            self.prev_gxy = (gp[0], gp[1]); self.prev_pxy = (peg_p[0], peg_p[1])

            if gz >= lift_goal[2] - 1e-3:
                self.phase = 4

        elif self.phase == 4:  # move above hole (slow) with grasp/slip checks
            grasp_ok = self._grasp_ok(f_gp_total, f_left, f_right, gp, peg_p)
            if not grasp_ok:
                self.lost_grasp_frames += 1
                target_pos = (gp[0], gp[1], gp[2] + 0.004)
                target_jaw = max(self.jaw_close - 0.0005, 0.0)
                if self.lost_grasp_frames >= self.lost_grasp_frames_thresh:
                    self.phase = 3
            else:
                self.lost_grasp_frames = 0
                target_pos = self._lerp3(gp, hole_above, self.travel_alpha_slow/10)
                target_jaw = self.jaw_close
                if self._dist3(gp, hole_above) < 0.01:
                    # init admittance state here
                    self.z_ref = gp[2]
                    self.v_z   = 0.0
                    self.timer = 0
                    self.phase = 5

            self.prev_gz = gp[2]; self.prev_pz = peg_p[2]
            self.prev_gxy = (gp[0], gp[1]); self.prev_pxy = (peg_p[0], peg_p[1])

        else:  # ---------------- PHASE 5: Admittance-based insertion ----------------
            # Measure contact normal & force between peg and cuboid
            F_meas, n_avg = self._contact_force_and_normal(self.pid, self.cid)
            n_xy = (n_avg[0], n_avg[1])
            n_xy_norm = math.hypot(n_xy[0], n_xy[1]) + 1e-9
            n_xy_unit = (n_xy[0]/n_xy_norm, n_xy[1]/n_xy_norm)

            # Force error (want +Fd along normal; PyBullet normal is on B; sign not critical here since we mod z only)
            eF = (self.Fd - F_meas)

            # Admittance update (discrete)
            # m*dv/dt + b*v + k*(z - z_ref) = eF  ->  v += dt/m * (eF - b*v - k*(z - z_ref))
            v_next = self.v_z + (self.dt / self.m_z) * (eF - self.b_z * self.v_z - self.k_z * (gp[2] - self.z_ref))
            # Clamp velocity to keep things gentle
            v_next = max(-self.vz_max, min(self.vz_max, v_next))

            # Proposed z target from admittance
            z_next = gp[2] + v_next * self.dt

            # If no contact yet, do a small constant search descent
            if F_meas < 0.5:
                z_next = gp[2] - self.insert_step * 0.5

            # Lateral compliance: move slightly opposite lateral component of the contact normal
            # (reduce side load). We bias toward hole XY center while compensating with normal.
            F_xy = min(F_meas, self.F_cap_xy)
            corr_x = -self.xy_beta * n_xy_unit[0] * (F_xy / self.F_cap_xy)
            corr_y = -self.xy_beta * n_xy_unit[1] * (F_xy / self.F_cap_xy)

            x_des = hole_p[0] + corr_x
            y_des = hole_p[1] + corr_y

            # Final target (slow tracking)
            target_pos = self._lerp3(gp, (x_des, y_des, z_next), self.travel_alpha_slow)
            target_jaw = self.jaw_close

            # Safety: if force spikes too high, micro back-off & increase damping transiently
            if F_meas > self.insert_force_max:
                target_pos = self._lerp3(gp, (hole_p[0], hole_p[1], gp[2] + 0.002), 0.5)
                # transient damping bump
                self.v_z *= 0.5

            # Update admittance state
            self.v_z = v_next
            # Optionally adapt z_ref slowly to follow settled depth
            self.z_ref = 0.999 * self.z_ref + 0.001 * gp[2]

        # --------- PRINT STATUS (phase, gripper pos, peg pos, forces) ---------
        print(f"[SimpleAgent] phase={self._phase_name(self.phase)} "
              f"gp=({gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}) "
              f"peg=({peg_p[0]:.3f}, {peg_p[1]:.3f}, {peg_p[2]:.3f}) "
              f"Fgp(total={f_gp_total:.2f}) Fpc={f_pc_total:.2f} vz={getattr(self,'v_z',0.0):.4f}")

        return target_pos, target_orn, target_jaw

    # ---------------- helpers ----------------
    def _phase_name(self, ph: int) -> str:
        return {
            0: "above_peg/open",
            1: "descend/open",
            2: "close_confirm",
            3: "lift_guarded",
            4: "move_above_hole",
            5: "insert_admittance",
        }.get(ph, f"unknown({ph})")

    def _grasp_ok(self, f_total, f_left, f_right, gp, peg_p) -> bool:
        # relaxed force + follow checks
        force_ok = (f_total >= self.force_hold_min) and \
                   (f_left >= 0.6*self.force_contact_min) and \
                   (f_right >= 0.6*self.force_contact_min)

        follow_ok = True
        if self.prev_gz is not None and self.prev_pz is not None:
            dzg = gp[2] - self.prev_gz
            dzp = peg_p[2] - self.prev_pz
            if dzg > 0 and (dzg - dzp) > self.lift_follow_tol:
                follow_ok = False

        if self.prev_gxy is not None and self.prev_pxy is not None:
            dgx = gp[0] - self.prev_gxy[0]; dgy = gp[1] - self.prev_gxy[1]
            dpx = peg_p[0] - self.prev_pxy[0]; dpy = peg_p[1] - self.prev_pxy[1]
            dxy_g = math.hypot(dgx, dgy); dxy_p = math.hypot(dpx, dpy)
            if (dxy_g - dxy_p) > self.xy_follow_tol:
                follow_ok = False

        return force_ok and follow_ok

    def _total_normal_force(self, a: int, b: int) -> float:
        return float(sum(cp[9] for cp in p.getContactPoints(bodyA=a, bodyB=b)))

    def _contact_force_and_normal(self, a: int, b: int):
        """
        Returns (total_normal_force, average_normal_world_on_b)
        PyBullet: cp[9] is Fn; cp[7] is contact normal on B (x,y,z) in world coords.
        """
        cps = p.getContactPoints(bodyA=a, bodyB=b)
        if not cps:
            return 0.0, (0.0, 0.0, 1.0)
        F = 0.0
        nx = ny = nz = 0.0
        for cp in cps:
            fn = cp[9]
            F += fn
            # average normals weighted by force
            n = cp[7] if isinstance(cp[7], (list, tuple)) else None
            if n is None:
                # Some builds flatten normal components as cp[7], cp[8], cp[9] but cp[9] is Fn we already used.
                # Try robust extraction using provided fields; fallback to zero lateral.
                n_world = p.getContactNormal(a, b) if hasattr(p, "getContactNormal") else (0.0, 0.0, 1.0)
                nx += n_world[0] * fn; ny += n_world[1] * fn; nz += n_world[2] * fn
            else:
                nx += n[0] * fn; ny += n[1] * fn; nz += n[2] * fn
        if F > 1e-9:
            return F, (nx/F, ny/F, nz/F)
        return 0.0, (0.0, 0.0, 1.0)

    def _gripper_peg_forces(self):
        """
        Return (total, left_pad, right_pad) normal forces between gripper and peg.
        If you know exact pad link indices, filter cp[3] accordingly.
        """
        left, right, total = 0.0, 0.0, 0.0
        for cp in p.getContactPoints(bodyA=self.gid, bodyB=self.pid):
            fn = cp[9]; total += fn
            # Heuristic split; replace with explicit {left_id, right_id} if known
            if cp[3] % 2 == 0: left = max(left, fn)
            else:               right = max(right, fn)
        return total, left, right

    @staticmethod
    def _dist3(a, b):
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

    @staticmethod
    def _lerp3(a, b, t):
        return (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*(b[2]-a[2]))
