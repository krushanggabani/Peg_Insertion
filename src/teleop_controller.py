import pybullet as p
import math
from typing import Iterable

# After p.connect(p.GUI):
# p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)

def _codes(chars: Iterable[str]):
    return [ord(c) for c in chars]

class KeyboardTeleop:
    """
    Translate:  s/x (±Y), z/c (±X), a/d (±Z)
    Rotate:     ← → (yaw), ↑ ↓ (pitch), q/e (roll)   [tries ',' '.' if delivered]
    Jaw:        u (close), o (open)                  [tries '[' ']' if delivered]
    Speed:      PAGE UP or 'k' (faster), PAGE DOWN or 'j' (slower)
    Fine:       hold SHIFT to halve translation step
    HUD:        red speed text near gripper
    """

    def __init__(self, step_pos=0.005, step_ang_deg=2.0, step_jaw_mm=1.0,
                 jaw_min_m=0.0, jaw_max_m=0.085,
                 dp_min=1e-4, dp_max=0.05, speed_scale=1.25,
                 show_hud=True, hud_color=(1.0, 0.0, 0.0), hud_offset=(0.0, 0.0, 0.10)):
        self.dp = float(step_pos)
        self.da = math.radians(step_ang_deg)
        self.dj = float(step_jaw_mm) / 1000.0
        self.dp_min, self.dp_max = float(dp_min), float(dp_max)
        self.speed_scale = float(speed_scale)
        self.jmin, self.jmax = float(jaw_min_m), float(jaw_max_m)
        self._jaw = float(jaw_max_m)

        self.show_hud = bool(show_hud)
        self.hud_color = tuple(hud_color)
        self.hud_offset = tuple(hud_offset)
        self._hud_uid = None
        self._last_speed_str = None

        # robust key sets (letters always work; punctuation only if delivered)
        self.keys_roll_plus  = set(_codes("q") + _codes(",<"))
        self.keys_roll_minus = set(_codes("e") + _codes(".>"))
        self.keys_jaw_close  = set(_codes("u") + _codes("[{"))
        self.keys_jaw_open   = set(_codes("o") + _codes("]}"))

        # speed (PAGE UP/DOWN + letter fallbacks)
        self.keys_speed_up   = {p.B3G_PAGE_UP,  ord('k')}
        self.keys_speed_down = {p.B3G_PAGE_DOWN, ord('j')}

        self.print_instructions()

    # ------- helpers -------
    @staticmethod
    def _is_down(keys, code):
        return (code in keys) and (keys[code] & p.KEY_IS_DOWN)

    @staticmethod
    def _triggered(keys, code):
        return (code in keys) and (keys[code] & p.KEY_WAS_TRIGGERED)

    @staticmethod
    def _down_any(keys, codes):
        for c in codes:
            if (c in keys) and (keys[c] & p.KEY_IS_DOWN):
                return True
        return False

    def _clamp_speed(self):
        self.dp = max(self.dp_min, min(self.dp, self.dp_max))

    def _update_speed_hud(self, anchor_pos):
        if not self.show_hud or anchor_pos is None:
            return
        label = f"speed: {self.dp:.4f} m/step"
        if label == self._last_speed_str and self._hud_uid is not None:
            return
        self._last_speed_str = label

        x, y, z = anchor_pos
        ox, oy, oz = self.hud_offset
        pos = [x + ox, y + oy, z + oz]

        if self._hud_uid is not None:
            p.removeUserDebugItem(self._hud_uid)
        self._hud_uid = p.addUserDebugText(
            label, textPosition=pos, textColorRGB=self.hud_color, textSize=1.4, lifeTime=0
        )

    def print_instructions(self):
        print(
            "[Teleop]\n"
            "  Translate : s/x (±Y), z/c (±X), a/d (±Z)\n"
            "  Rotate    : ← → (yaw), ↑ ↓ (pitch), q/e (roll)  [',' '.' if supported]\n"
            "  Jaw       : u (close), o (open)                 ['[' ']' if supported]\n"
            "  Speed     : PAGE UP or 'k' (faster), PAGE DOWN or 'j' (slower)\n"
            "  Fine      : hold SHIFT to halve translation step\n"
            "  NOTE      : Make sure the PyBullet window has focus.\n"
        )

    # ------- main -------
    def step(self, grip_pos, grip_orn):
        keys = p.getKeyboardEvents()

        # unpack
        px, py, pz = list(grip_pos)
        r, pch, yaw = p.getEulerFromQuaternion(grip_orn)

        # speed (use one-shot triggers)
        if any(self._triggered(keys, k) for k in self.keys_speed_up):
            self.dp *= self.speed_scale; self._clamp_speed()
        if any(self._triggered(keys, k) for k in self.keys_speed_down):
            self.dp /= self.speed_scale; self._clamp_speed()

        # SHIFT halves translational step
        dp_use = self.dp * (0.5 if ((p.B3G_SHIFT in keys) and (keys[p.B3G_SHIFT] & p.KEY_IS_DOWN)) else 1.0)

        # translation
        if self._is_down(keys, ord('s')): py += dp_use
        if self._is_down(keys, ord('x')): py -= dp_use
        if self._is_down(keys, ord('z')): px -= dp_use
        if self._is_down(keys, ord('c')): px += dp_use
        if self._is_down(keys, ord('a')): pz += dp_use
        if self._is_down(keys, ord('d')): pz -= dp_use

        # orientation
        if self._is_down(keys, p.B3G_LEFT_ARROW):   yaw += self.da
        if self._is_down(keys, p.B3G_RIGHT_ARROW):  yaw -= self.da
        if self._is_down(keys, p.B3G_UP_ARROW):     pch += self.da
        if self._is_down(keys, p.B3G_DOWN_ARROW):   pch -= self.da
        if self._down_any(keys, self.keys_roll_plus):   r += self.da
        if self._down_any(keys, self.keys_roll_minus):  r -= self.da

        # jaw
        if self._down_any(keys, self.keys_jaw_close):
            self._jaw = max(self.jmin, self._jaw - self.dj)
        if self._down_any(keys, self.keys_jaw_open):
            self._jaw = min(self.jmax, self._jaw + self.dj)

        # HUD
        self._update_speed_hud(anchor_pos=(px, py, pz))

        new_orn = p.getQuaternionFromEuler([r, pch, yaw])
        return (px, py, pz), new_orn, self._jaw
