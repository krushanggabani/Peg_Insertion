import pybullet as p
import math

class KeyboardTeleop:
    """
    Keys:
      Translate:  s/x (±Y), z/c (±X), a/d (±Z)
      Rotate:     ← → (yaw), ↑ ↓ (pitch), ',' '.' (roll)
      Jaw:        '[' close, ']' open
      Speed:      '=' faster, '-' slower   (one-shot)
      Fine ctrl:  hold SHIFT to halve translation step

    Notes:
      - Prints a controls cheat-sheet to the terminal once at init.
      - Shows current translation speed in the GUI near the gripper (red text).
    """
    def __init__(self, step_pos=0.005, step_ang_deg=2.0, step_jaw_mm=1.0,
                 jaw_min_m=0.0, jaw_max_m=0.085,
                 dp_min=1e-4, dp_max=0.05, speed_scale=1.25,
                 show_hud=True, hud_color=(1.0, 0.0, 0.0), hud_offset=(0.0, 0.0, 0.10)):
        # motion params
        self.dp = float(step_pos)
        self.da = math.radians(step_ang_deg)
        self.dj = float(step_jaw_mm) / 1000.0

        # speed constraints
        self.dp_min = float(dp_min)
        self.dp_max = float(dp_max)
        self.speed_scale = float(speed_scale)

        # gripper limits
        self.jmin = float(jaw_min_m)
        self.jmax = float(jaw_max_m)
        self._jaw = float(jaw_max_m)  # start open

        # HUD config
        self.show_hud = bool(show_hud)
        self.hud_color = tuple(hud_color)
        self.hud_offset = tuple(hud_offset)
        self._hud_uid = None
        self._last_speed_str = None

        # print instructions once
        self.print_instructions()

    # ---------- public helpers ----------
    def print_instructions(self):
        msg = """
[Teleop]
  Translate : s/x (±Y), z/c (±X), a/d (±Z)
  Rotate    : ← → (yaw), ↑ ↓ (pitch), ',' '.' (roll)
  Jaw       : '[' close, ']' open
  Speed     : '=' faster, '-' slower   (one-shot)
  Fine ctrl : hold SHIFT to halve translation step

  Tip: current speed is shown in the GUI near the gripper.
"""
        print(msg.strip())

    # ---------- internal helpers ----------
    @staticmethod
    def _is_down(keys, k):
        return (k in keys) and (keys[k] & p.KEY_IS_DOWN)

    @staticmethod
    def _triggered(keys, k):
        return (k in keys) and (keys[k] & p.KEY_WAS_TRIGGERED)

    def _clamp_speed(self):
        self.dp = max(self.dp_min, min(self.dp, self.dp_max))

    def _update_speed_hud(self, anchor_pos):
        if not self.show_hud or anchor_pos is None:
            return
        text = f"speed: {self.dp:.4f} m/step"
        if text == self._last_speed_str and self._hud_uid is not None:
            return  # avoid redundant GUI updates
        self._last_speed_str = text

        x, y, z = anchor_pos
        ox, oy, oz = self.hud_offset
        pos = [x + ox, y + oy, z + oz]

        # replace previous label
        if self._hud_uid is not None:
            p.removeUserDebugItem(self._hud_uid)
        self._hud_uid = p.addUserDebugText(
            text,
            textPosition=pos,
            textColorRGB=self.hud_color,  # RED by default
            textSize=1.4,
            lifeTime=0   # persistent until removed/overwritten
        )

    # ---------- main step ----------
    def step(self, grip_pos, grip_orn):
        keys = p.getKeyboardEvents()

        # unpack
        px, py, pz = list(grip_pos)
        r, pch, yaw = p.getEulerFromQuaternion(grip_orn)

        # one-shot speed tweaks (common keyboards map '+' to '=' with shift)
        if self._triggered(keys, ord('=')):   # faster
            self.dp *= self.speed_scale
            self._clamp_speed()
        if self._triggered(keys, ord('-')):   # slower
            self.dp /= self.speed_scale
            self._clamp_speed()

        # “fine” control if SHIFT held (halve dp for this step)
        dp_use = self.dp
        if self._is_down(keys, p.B3G_SHIFT):
            dp_use *= 0.5

        # translation (your mapping kept)
        if self._is_down(keys, ord('s')): py += dp_use   # +Y
        if self._is_down(keys, ord('x')): py -= dp_use   # -Y
        if self._is_down(keys, ord('z')): px -= dp_use   # -X
        if self._is_down(keys, ord('c')): px += dp_use   # +X
        if self._is_down(keys, ord('a')): pz += dp_use   # +Z
        if self._is_down(keys, ord('d')): pz -= dp_use   # -Z

        # orientation
        if self._is_down(keys, p.B3G_LEFT_ARROW):   yaw += self.da
        if self._is_down(keys, p.B3G_RIGHT_ARROW):  yaw -= self.da
        if self._is_down(keys, p.B3G_UP_ARROW):     pch += self.da
        if self._is_down(keys, p.B3G_DOWN_ARROW):   pch -= self.da
        if self._is_down(keys, ord(',')):           r   += self.da
        if self._is_down(keys, ord('.')):           r   -= self.da

        # jaw
        if self._is_down(keys, ord('[')):
            self._jaw = max(self.jmin, self._jaw - self.dj)
        if self._is_down(keys, ord(']')):
            self._jaw = min(self.jmax, self._jaw + self.dj)

        # GUI HUD update (anchor near gripper)
        self._update_speed_hud(anchor_pos=(px, py, pz))

        new_orn = p.getQuaternionFromEuler([r, pch, yaw])
        return (px, py, pz), new_orn, self._jaw

