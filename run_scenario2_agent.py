# run_scenario2_agent.py
import time
import pybullet as p

from env_peg_insertion import EnvConfig, PegInsertionEnv
from controller2 import SimpleAgent  # your force-aware agent

ASSETS_ROOT = "/home/krushang/Desktop/Research/OLD/Extra_env/Bullet_Project/assets"

cfg = EnvConfig(
    assets_root=ASSETS_ROOT,
    gui=True,
    real_time=False,          # agent can run faster/slower with fixed time step
    time_step=1.0 / 240.0,
    log_path="logs/scenario2_agent.csv"
)

# Build environment
env = PegInsertionEnv(cfg)
env.reset()

# Instantiate the force-aware agent with body IDs
agent = SimpleAgent(
    cfg,
    gripper_id=env.gripper.id,   # gripper body unique ID from the env
    peg_id=env.peg_id,           # peg body ID
    cuboid_id=env.cuboid_id      # cuboid body ID
)
agent.reset()

def hole_pose_world(env):
    """
    Convert the cuboid's hole center (given in cuboid local frame via cfg.hole_center)
    into world coordinates.
    """
    cub_pos, cub_orn = p.getBasePositionAndOrientation(env.cuboid_id)
    R = p.getMatrixFromQuaternion(cub_orn)
    R = [[R[0], R[1], R[2]],
         [R[3], R[4], R[5]],
         [R[6], R[7], R[8]]]
    hc = env.cfg.hole_center
    off = [R[0][0]*hc[0] + R[0][1]*hc[1] + R[0][2]*hc[2],
           R[1][0]*hc[0] + R[1][1]*hc[1] + R[1][2]*hc[2],
           R[2][0]*hc[0] + R[2][1]*hc[1] + R[2][2]*hc[2]]
    return ([cub_pos[0] + off[0], cub_pos[1] + off[1], cub_pos[2] + off[2]], cub_orn)

max_steps = 60000

try:
    for _ in range(max_steps):
        obs = env.get_obs()
        hp = hole_pose_world(env)

        # Force-aware agent produces (pos, orn, jaw) commands
        cmd_pos, cmd_orn, cmd_jaw = agent.act(obs, hp)

        # Step the env with commanded pose & jaw; physics runs inside
        env.step(cmd_pos, cmd_orn, cmd_jaw, do_sim=True)

        # Success condition from env (precise clearance check is inside env)
        if env.is_inserted():
            print("[SUCCESS] Peg inserted by agent!")
            # Show a brief success label at current gripper pose
            p.addUserDebugText(
                "SUCCESS: Peg inserted",
                textPosition=[cmd_pos[0], cmd_pos[1], cmd_pos[2] + 0.08],
                textColorRGB=[0.2, 1.0, 0.3],
                textSize=1.8,
                lifeTime=2.0,
            )
            time.sleep(0.25)
            break

        # For fully accelerated runs set this to 0.0; keep small sleep for nicer visuals
        time.sleep(1 / 240)

finally:
    env.close()
    print(f"Logs written to {cfg.log_path}")
