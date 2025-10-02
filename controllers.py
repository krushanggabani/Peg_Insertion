# controllers.py
import pybullet as p
import time
import math

class KeyboardTeleop:
    """
    WASD / RF translate, arrow keys yaw/pitch, ',' '.' roll,
    '[' ']' for jaw separation in mm steps.
    """
    def __init__(self, step_pos=0.005, step_ang_deg=2.0, step_jaw_mm=1.0,
                 jaw_min_m=0.0, jaw_max_m=0.085):
        self.dp = step_pos
        self.da = math.radians(step_ang_deg)
        self.dj = step_jaw_mm / 1000.0
        self.jmin = jaw_min_m
        self.jmax = jaw_max_m
        self._jaw = 0.085

    def step(self, grip_pos, grip_orn):
        keys = p.getKeyboardEvents()

        # unpack
        px, py, pz = list(grip_pos)
        ex, ey, ez, ew = grip_orn
        r, pch, y = p.getEulerFromQuaternion(grip_orn)

        # translation
        if ord('w') in keys: py += self.dp
        if ord('s') in keys: py -= self.dp
        if ord('a') in keys: px -= self.dp
        if ord('d') in keys: px += self.dp
        if ord('r') in keys: pz += self.dp
        if ord('f') in keys: pz -= self.dp

        # orientation
        if p.B3G_LEFT_ARROW in keys:  y += self.da
        if p.B3G_RIGHT_ARROW in keys: y -= self.da
        if p.B3G_UP_ARROW in keys:    pch += self.da
        if p.B3G_DOWN_ARROW in keys:  pch -= self.da
        if ord(',') in keys:          r += self.da
        if ord('.') in keys:          r -= self.da

        # jaw
        if ord('[') in keys: self._jaw = max(self.jmin, self._jaw - self.dj)
        if ord(']') in keys: self._jaw = min(self.jmax, self._jaw + self.dj)

        new_orn = p.getQuaternionFromEuler([r, pch, y])
        return (px, py, pz), new_orn, self._jaw


class SimpleAgent:
    """A super-simple scripted agent for Scenario 2 to validate the pipeline."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = 0
        self.timer = 0

    def reset(self):
        self.phase = 0
        self.timer = 0

    def act(self, obs, hole_pose_world):
        """
        hole_pose_world: (pos_xyz, orn_xyzw) of cuboid hole center in world
        Strategy:
          0 open near peg
          1 descend align
          2 close
          3 lift
          4 move above hole
          5 insert
        """
        gp = obs["grip_pos"]; go = obs["grip_orn"]; jaw = obs["grip_jaw_sep"]
        peg_p = obs["peg_pos"]; peg_o = obs["peg_orn"]

        # defaults
        target_pos = list(gp)
        target_orn = go
        target_jaw = 0.085

        gripper_open  = 0.010
        gripper_close = 0.085 

        # set some waypoints
        peg_above = (peg_p[0], peg_p[1], peg_p[2] + 0.2)
        hole_p, hole_o = hole_pose_world
        hole_above = (hole_p[0], hole_p[1], hole_p[2] + 0.06)

        if self.phase == 0:  # move above peg, open
            target_pos = lerp3(gp, peg_above, 0.15)
            target_jaw = gripper_open
            if dist3(gp, peg_above) < 0.01:
                self.phase = 1

        elif self.phase == 1:  # descend to peg
            target_pos = lerp3(gp, (peg_p[0], peg_p[1], peg_p[2]+0.135), 0.2)
            target_jaw = gripper_open
            if abs(gp[2] - (peg_p[2]+0.135)) < 0.003:
                self.phase = 2

        elif self.phase == 2:  # close
            target_pos = gp
            target_jaw = gripper_close
            self.timer += 1
            if self.timer > 120:
                self.timer = 0
                self.phase = 3

        elif self.phase == 3:  # lift peg
            target_pos = lerp3(gp, (gp[0], gp[1], max(gp[2], peg_p[2]+0.2)), 0.2)
            target_jaw = gripper_close
            if gp[2] >= peg_p[2]+0.075:
                self.phase = 3

        elif self.phase == 4:  # move above hole
            target_pos = lerp3(gp, hole_above, 0.15)
            target_jaw = 0.010
            if dist3(gp, hole_above) < 0.01:
                self.phase = 5

        else:  # insert
            target_pos = lerp3(gp, (hole_p[0], hole_p[1], hole_p[2]+0.010), 0.10)
            target_jaw = 0.010

        return target_pos, target_orn, target_jaw


def dist3(a, b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def lerp3(a, b, t):
    return (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*(b[2]-a[2]))
