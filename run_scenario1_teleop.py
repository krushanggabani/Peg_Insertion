# run_scenario1_teleop.py
import time
import pybullet as p

from env_peg_insertion import EnvConfig, PegInsertionEnv
from controllers import KeyboardTeleop

# from your_module import Robotiq85, load_mesh

ASSETS_ROOT = "/home/krushang/Desktop/Research/OLD/Extra_env/Bullet_Project/assets"
GRIP_URDF   = "/home/krushang/Desktop/Research/Roboforce/Peg_Insertion/assets/urdf/robotiq_85.urdf"

# If your Robotiq85 class uses a hardcoded path, you’re fine; otherwise ensure it points to GRIP_URDF.
from utils import Robotiq85, load_mesh  # adjust import to your file structure

cfg = EnvConfig(
    assets_root=ASSETS_ROOT,
    gui=True,
    real_time=True,   # tele-op is easier in real-time
    log_path="logs/scenario1_teleop.csv"
)

env = PegInsertionEnv(cfg, Robotiq85, load_mesh)
env.reset()

teleop = KeyboardTeleop(
    step_pos=0.004, step_ang_deg=2.0,
    step_jaw_mm=0.8,
    jaw_min_m=cfg.jaw_closed_m, jaw_max_m=cfg.jaw_open_m
)

print("[Tele-Op] Controls: WASD/R/F move, arrows & , . rotate, [ ] jaw, ESC to quit.")
try:
    while p.isConnected():
        obs = env.get_obs()
        cmd_pos, cmd_orn, cmd_jaw = teleop.step(obs["grip_pos"], obs["grip_orn"])
        env.step(cmd_pos, cmd_orn, cmd_jaw, do_sim=True)

        if env.is_inserted():
            print("[SUCCESS] Peg inserted!")
            break

        # small sleep just to be nice on CPU (real_time sim is already pacing)
        time.sleep(1/240.0)

finally:
    env.close()
    print(f"Logs written to {cfg.log_path}")
