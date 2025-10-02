# run_scenario2_agent.py
import time
import pybullet as p

from env_peg_insertion import EnvConfig, PegInsertionEnv
from controllers import SimpleAgent

# from your_module import Robotiq85, load_mesh

ASSETS_ROOT = "/home/krushang/Desktop/Research/OLD/Extra_env/Bullet_Project/assets"
from utils import Robotiq85, load_mesh  # adjust import to your file structure

cfg = EnvConfig(
    assets_root=ASSETS_ROOT,
    gui=True,
    real_time=False,   # agent can run faster/slower with fixedTimeStep
    time_step=1.0/240.0,
    log_path="logs/scenario2_agent.csv"
)
env = PegInsertionEnv(cfg, Robotiq85, load_mesh)
env.reset()

agent = SimpleAgent(cfg)
agent.reset()

def hole_pose_world(env):
    # Convert cuboid hole center (in cuboid local frame) to world
    cub_pos, cub_orn = p.getBasePositionAndOrientation(env.cuboid_id)
    R = p.getMatrixFromQuaternion(cub_orn)
    R = [[R[0],R[1],R[2]], [R[3],R[4],R[5]], [R[6],R[7],R[8]]]
    hc = env.cfg.hole_center
    off = [R[0][0]*hc[0] + R[0][1]*hc[1] + R[0][2]*hc[2],
           R[1][0]*hc[0] + R[1][1]*hc[1] + R[1][2]*hc[2],
           R[2][0]*hc[0] + R[2][1]*hc[1] + R[2][2]*hc[2]]
    return ([cub_pos[0]+off[0], cub_pos[1]+off[1], cub_pos[2]+off[2]], cub_orn)

max_steps = 60000
try:
    for _ in range(max_steps):
        obs = env.get_obs()
        hp = hole_pose_world(env)
        cmd_pos, cmd_orn, cmd_jaw = agent.act(obs, hp)
        env.step(cmd_pos, cmd_orn, cmd_jaw, do_sim=True)

        if env.is_inserted():
            print("[SUCCESS] Peg inserted by agent!")
            break

        time.sleep(1/240)  # no wait (accelerated). set >0.0 to slow down visually

finally:
    env.close()
    print(f"Logs written to {cfg.log_path}")
