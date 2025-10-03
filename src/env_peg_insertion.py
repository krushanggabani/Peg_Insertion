# env_peg_insertion.py
import os
import csv
import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict

import pybullet as p
import pybullet_data
import numpy as np

from src.recorder import PBRecorder           # <-- integrated
from src.utils import Robotiq85, load_mesh    # adjust import to your file structure


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
    real_time: bool = False                  # tele-op: True
    solver_iters: int = 150
    contact_erp: float = 0.2
    contact_slop: float = 0.001
    lateral_friction: float = 1.0
    spinning_friction: float = 0.001
    rolling_friction: float = 0.0
    restitution: float = 0.0

    # hole goal (cuboid local frame)
    hole_center: Tuple[float,float,float] = (0.0, 0.0, 0.05)
    hole_half_extents: Tuple[float,float,float] = (0.006, 0.006, 0.015)
    max_ori_tilt_deg: float = 7.0

    # jaw separation ↔ joint angle mapping
    jaw_open_m: float = 0.085
    jaw_closed_m: float = 0.0
    joint_open_rad: float = 0.85
    joint_closed_rad: float = 0.0

    # initial placement
    peg_start: Tuple[float,float,float] = (0.0, 0.20, 0.1)
    peg_orn  : Tuple[float,float,float] = (0,0,0)           # Euler; converted below
    cuboid_start: Tuple[float,float,float] = (0.0, -0.20, 0.05)
    gripper_start: Tuple[float,float,float] = (0.0, 0.50, 0.20)
    gripper_start_rpy: Tuple[float,float,float] = (math.pi, 0.0, 0.0)

    # logging
    log_path: str = "logs/peg_run.csv"

    # --- Recording ---
    record_video: bool = True                 # MP4 via GUI logger
    record_gif: bool = True                   # GIF via offscreen camera grab
    video_path: str = "logs/scenario1.mp4"
    gif_path: str = "logs/scenario1.gif"
    gif_fps: int = 20
    gif_size: Tuple[int,int] = (640, 480)
    gif_stride: int = 20                      # <-- NEW: save every Nth frame (both GIF & MP4 fallback)

    # Camera for GIF (if None, uses debug cam)
    cam_target: Tuple[float,float,float] = (0.5, 0.0, 0.6)
    cam_distance: float = 1.0
    cam_yaw: float = 90.0
    cam_pitch: float = -35.0
    cam_fov_deg: float = 45.0
    cam_near: float = 0.01
    cam_far: float = 5.0


class PegInsertionEnv:
    """
    - Kinematic gripper base; physical links for contact.
    - Robotiq 85 jaws via position control on parent joint + gear mimic constraints.
    - Logs commands, actuals, forces each step.
    - Built-in video/GIF recorder managed by the env lifecycle.
    """
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._robotiq_cls = Robotiq85
        self._load_mesh_fn = load_mesh

        # --- Connect & configure sim ---
        self.client = p.connect(p.GUI if cfg.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
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

        # --- Gripper ---
        self.gripper = self._robotiq_cls(cfg.gripper_start, cfg.gripper_start_rpy)
        self.gripper.load()

        # increase pad friction (adjust link ids to your URDF)
        for link_id in [3, 8]:
            p.changeDynamics(self.gripper.id, link_id,
                             lateralFriction=1000.0,
                             spinningFriction=1.0,
                             frictionAnchor=1)

        # --- Objects ---
        self.cuboid_id = load_mesh_fn_rel(self._load_mesh_fn, cfg.assets_root, "cuboid.stl",
                                          pos=cfg.cuboid_start, mass=0.0, mesh_scale=0.001, fixed=False)

        cfg.peg_orn = p.getQuaternionFromEuler(cfg.peg_orn)
        self.peg_id = load_mesh_fn_rel(self._load_mesh_fn, cfg.assets_root, "peg.stl",
                                       pos=cfg.peg_start, orn=cfg.peg_orn, mass=0.35, mesh_scale=0.001, fixed=False)

        # GUI camera (only affects GUI, not GIF)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.00,
            cameraYaw=91,
            cameraPitch=-30.60,
            cameraTargetPosition=(-1.19, 0.00, -0.66)
        )

        # dynamics on all
        for body in [self.gripper.id, self.cuboid_id, self.peg_id]:
            self._apply_dynamics(body)

        # --- Logging ---
        os.makedirs(os.path.dirname(cfg.log_path) or ".", exist_ok=True)
        self.log_fp = open(cfg.log_path, "w", newline="")
        self.logger = csv.writer(self.log_fp)
        self.logger.writerow(DEFAULT_LOG_HEADERS)

        # --- Recorder ---
        self.rec = PBRecorder()
        if self.cfg.record_video and self.cfg.gui:
            os.makedirs(os.path.dirname(self.cfg.video_path) or ".", exist_ok=True)
            self.rec.start_video(self.cfg.video_path, fps=60)
        if self.cfg.record_gif:
            os.makedirs(os.path.dirname(self.cfg.gif_path) or ".", exist_ok=True)
            self.rec.start_gif(
                path=self.cfg.gif_path,
                fps=self.cfg.gif_fps,
                size=self.cfg.gif_size,
                target=self.cfg.cam_target,
                distance=self.cfg.cam_distance,
                yaw=self.cfg.cam_yaw,
                pitch=self.cfg.cam_pitch,
                fov_deg=self.cfg.cam_fov_deg,
                near=self.cfg.cam_near,
                far=self.cfg.cam_far,
            )

        # Internals
        self.ticks = 0
        self._last_cmd = None  # (pos, orn, jaw_sep_m)

    # ---------- public API ----------
    def reset(self):
        # base poses
        p.resetBasePositionAndOrientation(self.gripper.id,
            self.cfg.gripper_start,
            p.getQuaternionFromEuler(self.cfg.gripper_start_rpy))
        p.resetBasePositionAndOrientation(self.cuboid_id, self.cfg.cuboid_start, [0,0,0,1])
        p.resetBasePositionAndOrientation(self.peg_id, self.cfg.peg_start, self.cfg.peg_orn)

        # open gripper
        self.command_jaw_separation(self.cfg.jaw_open_m)

        # settle (record only every Nth frame)
        for i in range(120):
            p.stepSimulation()
            if self.cfg.record_gif and (i % self.cfg.gif_stride == 0):
                self.rec.grab_frame()

        self.ticks = 0

    def close(self):
        # stop/save recordings
        try:
            if self.cfg.record_video and self.cfg.gui:
                self.rec.stop_video()
        except Exception:
            pass
        try:
            if self.cfg.record_gif:
                self.rec.save_gif()
        except Exception:
            pass

        # close log & sim
        try:
            self.log_fp.close()
        except Exception:
            pass
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def command_base_pose(self, pos_xyz, orn_xyzw):
        p.resetBasePositionAndOrientation(self.gripper.id, pos_xyz, orn_xyzw)

    def command_jaw_separation(self, jaw_sep_m: float):
        a0, a1 = self.cfg.joint_closed_rad, self.cfg.joint_open_rad
        s0, s1 = self.cfg.jaw_closed_m,     self.cfg.jaw_open_m
        t = (jaw_sep_m - s0) / max(1e-9, (s1 - s0))
        t = max(0.0, min(1.0, t))
        target_angle = a0 + t*(a1 - a0)
        self.gripper.move_gripper(target_angle)

    def step(self,
             cmd_pos_xyz: Tuple[float,float,float],
             cmd_orn_xyzw: Tuple[float,float,float,float],
             cmd_jaw_sep_m: float,
             do_sim: bool = True) -> Dict:
        self._last_cmd = (cmd_pos_xyz, cmd_orn_xyzw, cmd_jaw_sep_m)

        self.command_base_pose(cmd_pos_xyz, cmd_orn_xyzw)
        self.command_jaw_separation(cmd_jaw_sep_m)

        if do_sim and not self.cfg.real_time:
            p.stepSimulation()

        # record a frame only every Nth tick (applies to GIF & MP4 fallback)
        if self.cfg.record_gif and (self.ticks % self.cfg.gif_stride == 0):
            self.rec.grab_frame()

        self._log_once()
        self.ticks += 1
        return self.get_obs()

    # ---------- observation & logging ----------
    def get_obs(self) -> Dict:
        grip_pos, grip_orn = p.getBasePositionAndOrientation(self.gripper.id)
        peg_pos, peg_orn   = p.getBasePositionAndOrientation(self.peg_id)
        parent_angle = p.getJointState(self.gripper.id, self.gripper.mimic_parent_idx)[0]
        jaw_sep_est = self._angle_to_sep(parent_angle)
        return dict(
            grip_pos=grip_pos,
            grip_orn=grip_orn,
            grip_jaw_sep=jaw_sep_est,
            peg_pos=peg_pos,
            peg_orn=peg_orn
        )

    def _log_once(self):
        if self._last_cmd is None:
            return
        (cpos, corn, csep) = self._last_cmd

        grip_pos, grip_orn = p.getBasePositionAndOrientation(self.gripper.id)
        parent_angle = p.getJointState(self.gripper.id, self.gripper.mimic_parent_idx)[0]
        jaw_sep_est = self._angle_to_sep(parent_angle)
        peg_pos, peg_orn = p.getBasePositionAndOrientation(self.peg_id)

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
        if ori_deg_tol is None:
            ori_deg_tol = self.cfg.max_ori_tilt_deg

        peg_pos, peg_orn = p.getBasePositionAndOrientation(self.peg_id)
        cub_pos, cub_orn = p.getBasePositionAndOrientation(self.cuboid_id)

        peg_in_world = np.array(peg_pos)
        cub_pos = np.array(cub_pos)
        cub_R = np.array(p.getMatrixFromQuaternion(cub_orn)).reshape(3,3)
        peg_local = cub_R.T @ (peg_in_world - cub_pos)

        hc = np.array(self.cfg.hole_center)
        he = np.array(self.cfg.hole_half_extents)
        within = np.all(np.abs(peg_local - hc) <= (he + pos_tol))

        peg_R = np.array(p.getMatrixFromQuaternion(peg_orn)).reshape(3,3)
        peg_z_world = peg_R[:,2]
        cub_z_world = cub_R[:,2]
        ang = math.degrees(math.acos(np.clip(np.dot(peg_z_world, cub_z_world), -1.0, 1.0)))
        ok_ori = (abs(ang) <= ori_deg_tol)

        in_contact = 1
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
                contactStiffness=1e6, contactDamping=1e3
            )

    def _sum_contact_force(self, a: int, b: int) -> float:
        return float(sum(cp[9] for cp in p.getContactPoints(bodyA=a, bodyB=b)))

    def _angle_to_sep(self, angle_rad: float) -> float:
        a0, a1 = self.cfg.joint_closed_rad, self.cfg.joint_open_rad
        s0, s1 = self.cfg.jaw_closed_m,     self.cfg.jaw_open_m
        t = (angle_rad - a0) / max(1e-9, (a1 - a0))
        t = max(0.0, min(1.0, t))
        return float(s0 + t*(s1 - s0))


def load_mesh_fn_rel(load_mesh_fn, root, name, **kwargs):
    path = os.path.join(root, name)
    return load_mesh_fn(path, **kwargs)
