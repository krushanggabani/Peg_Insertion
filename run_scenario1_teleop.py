# run_scenario1_teleop.py
import time
import pybullet as p

from env_peg_insertion import EnvConfig, PegInsertionEnv
from controllers import KeyboardTeleop

ASSETS_ROOT = "/home/krushang/Desktop/Research/OLD/Extra_env/Bullet_Project/assets"
GRIP_URDF   = "/home/krushang/Desktop/Research/Roboforce/Peg_Insertion/assets/urdf/robotiq_85.urdf"




def set_camera_on_scene(env, extra_up=0.08, yaw=35, pitch=-35, distance=0.35):
    """Aim camera at the midpoint between cuboid and peg, slightly above."""
    try:
        peg_pos, _ = p.getBasePositionAndOrientation(env.peg_id)
        cub_pos, _ = p.getBasePositionAndOrientation(env.cuboid_id)
        target = [
            0.5 * (peg_pos[0] + cub_pos[0]),
            0.5 * (peg_pos[1] + cub_pos[1]),
            0.5 * (peg_pos[2] + cub_pos[2]) + extra_up,
        ]
    except Exception:
        # fallback: table-ish view
        target = [0.0, 0.0, 0.1 + extra_up]

    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
    )


def show_controls_hud(anchor=(0.0, 0.0, 0.35), line_spacing=0.035):
    """Draw a small controls cheat-sheet in the GUI (3D text near the scene)."""
    lines = [
        "Teleop Controls",
        "Move:  s/x (±Y), z/c (±X), a/d (±Z)",
        "Rotate: ← → (yaw), ↑ ↓ (pitch), ',' '.' (roll)",
        "Jaw:   '[' close, ']' open",
        "Speed: '=' faster, '-' slower  |  SHIFT = fine",
    ]
    color_title = [0.9, 0.9, 0.1]
    color_text  = [0.8, 0.9, 0.9]
    for i, text in enumerate(lines):
        color = color_title if i == 0 else color_text
        p.addUserDebugText(
            text,
            textPosition=[anchor[0], anchor[1], anchor[2] + (len(lines) - i) * line_spacing],
            textColorRGB=color,
            textSize=1.3,
            lifeTime=0,  # persistent
        )


cfg = EnvConfig(
    assets_root=ASSETS_ROOT,
    gui=True,
    real_time=True,   # tele-op is easier in real-time
    log_path="logs/scenario1_teleop.csv"
)

env = PegInsertionEnv(cfg)
env.reset()

# Make the GUI cleaner and camera closer
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide side panels
set_camera_on_scene(env, extra_up=0.08, yaw=35, pitch=-35, distance=0.35)
# Anchor the HUD roughly above the cuboid; tweak if needed
try:
    cub_pos, _ = p.getBasePositionAndOrientation(env.cuboid_id)
    show_controls_hud(anchor=[cub_pos[0] - 0.10, cub_pos[1] - 0.12, cub_pos[2] + 0.02])
except Exception:
    show_controls_hud()

teleop = KeyboardTeleop(
    step_pos=0.0005, step_ang_deg=2.0,
    step_jaw_mm=0.8,
    jaw_min_m=cfg.jaw_closed_m, jaw_max_m=cfg.jaw_open_m,
    show_hud=True,   # shows live speed near the gripper
)

try:
    while p.isConnected():
        obs = env.get_obs()
        cmd_pos, cmd_orn, cmd_jaw = teleop.step(obs["grip_pos"], obs["grip_orn"])
        env.step(cmd_pos, cmd_orn, cmd_jaw, do_sim=True)

        if env.is_inserted():
            print("[SUCCESS] Peg inserted!")
            # Briefly leave a success label
            p.addUserDebugText(
                "SUCCESS: Peg inserted",
                textPosition=[cmd_pos[0], cmd_pos[1], cmd_pos[2] + 0.08],
                textColorRGB=[0.2, 1.0, 0.3],
                textSize=1.8,
                lifeTime=2.0,
            )
            time.sleep(0.5)
            break

        # small sleep just to be nice on CPU (real_time sim is already pacing)
        time.sleep(1/240.0)

finally:
    env.close()
    print(f"Logs written to {cfg.log_path}")
