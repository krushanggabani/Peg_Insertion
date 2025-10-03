# Peg Insertion in PyBullet (Robotiq 2F‑85)

> Rectangular **peg‑in‑hole** task implemented in PyBullet with two runnable scenarios built on a shared environment (`PegInsertionEnv`). Includes **CSV logging**, optional **MP4/GIF** capture, collision‑shape utilities for **STL with a hole**, and handy debug visualizations.
>
> * **Scenario 1 — Tele‑op:** drive the gripper with the keyboard.
> * **Scenario 2 — Agent:** a scripted, **force‑aware** agent runs an **admittance‑controlled** insertion.



## Table of Contents

* [Quickstart](#quickstart)
* [Project Layout](#project-layout)
* [Run the Scenarios](#run-the-scenarios)

  * [Scenario 1 — Tele‑op](#scenario-1--tele-op)
  * [Scenario 2 — Agent](#scenario-2--agent)
* [Environment & Configuration](#environment--configuration)
* [Logging (CSV Schema)](#logging-csv-schema)
* [Common Issues & Fixes](#common-issues--fixes)
* [Tips & Tuning](#tips--tuning)
* [Development Notes](#development-notes)
* [License](#license)



## Quickstart

### Dependencies

```bash
python -V         # Python 3.9+ recommended
pip install pybullet numpy

# Optional (for recording GIF/MP4 if ffmpeg is not on PATH):
pip install imageio imageio-ffmpeg
# For best MP4 quality in GUI mode, having system `ffmpeg` on PATH is preferred.
# If system ffmpeg is missing, recording falls back to imageio-ffmpeg.
```

### Run from repo root

```bash
# Tele‑op (keyboard control)
python run_scenario_1_teleop.py

# Agent (scripted admittance controller)
python run_scenario_2_agent.py
```




## Project Layout

```text
your_repo/
├─ run_scenario_1_teleop.py   # entrypoint for Scenario 1
├─ run_scenario_2_agent.py    # entrypoint for Scenario 2
├─ assets/
│  ├─ urdf/robotiq_85.urdf
│  ├─ peg.stl
│  └─ cuboid.stl
├─ logs/                         # created at runtime
└─ src/
   ├─ env_peg_insertion.py       # environment, physics, success checks, logging, 
   ├─ recorder.py                # unified recording helper (MP4/GIF)
   ├─ simple_controller.py       # multi‑phase force‑aware agent + admittance 
   ├─ teleop_controller.py       # keyboard tele‑op + HUD
   ├─ utils.py                   # Robotiq85 mimic joints + mesh loader helpers

```

> **Note:** Meshes are imported with `mesh_scale=0.001` (millimeters → meters).



## Run the Scenarios

### Scenario 1 — Tele‑op

**What it does**

* Starts the env in real‑time **GUI** for smoother tele‑op.
* Shows a HUD (on‑screen cheat‑sheet), aims the camera to the scene, and loops until success/exit.
* Logs to `logs/scenario1_teleop.csv`.

**Controls** (focus the PyBullet window):

| Action    | Keys                                           |
| --------- | ---------------------------------------------- |
| Translate | `z`/`c` (±X), `s`/`x` (±Y), `a`/`d` (±Z)       |
| Rotate    | `←` `→` (yaw), `↑` `↓` (pitch), `q`/`e` (roll) |
| Jaw       | `u` close, `o` open                            |
| Speed     | `k` faster, `j` slower; fine tuning: hold **SHIFT**   |


HUD displays current step size.

**Success condition**

* Peg tip inside the hole AABB (cuboid local frame) **and** peg Z‑axis tilt ≤ `max_ori_tilt_deg`.
* A banner `SUCCESS: Peg inserted` appears on success.

---

### Scenario 2 — Agent

**What it does** (fixed `time_step`, default 1/240 s):

1. Move above peg (open)
2. Descend
3. Close & **confirm grasp by force**
4. Lift while monitoring slip
5. Move above hole with grasp checks
6. **Admittance‑based insertion** (lateral compliance, spike handling)

**Admittance loop (Phase 6)**

The vertical motion obeys a virtual mass‑damper‑spring with force feedback:

[ m \dot v + b v + k (z - z_{ref}) = F_d - F_{meas} ]

* Vertical velocity is clamped; lateral corrections use average contact normals to reduce side loads.
* Force limits guard against spikes.

**Logging**

* Writes `logs/scenario2_agent.csv` every step: commands, actual states, and summed contact forces.



## Environment & Configuration

```
env = PegInsertionEnv(cfg)
```

**Key `EnvConfig` fields (tunable)**

| Group     | Field(s)                                                               | Notes                                                |
| --------- | ---------------------------------------------------------------------- | ---------------------------------------------------- |
| Physics   | `time_step`, `solver_iters`, `contact_erp`, `frictions`, `restitution` | Global sim tuning                                    |
| Hole      | `hole_center`, `hole_half_extents`, `max_ori_tilt_deg`                 | AABB in cuboid local frame                           |
| Jaw       | linear map **jaw‑sep (m)** ↔ parent joint angle (rad)                  | Robotiq mimic joint mapping                          |
| Poses     | `peg_start`, `cuboid_start`, `gripper_start`, `gripper_start_rpy`      | Initial scene                                        |
| Recording | `record_video`, `record_gif`, `gif_stride`, paths & camera             | MP4 via GUI or GIF via imageio                       |
| Logging   | `log_path`                                                             | CSV with commands, actuals, peg pose, contact forces |

**Gripper & assets**

* Robotiq‑85 is implemented with **gear‑based mimic joints**; `move_gripper(angle_rad)` controls aperture.
* URDF: `assets/urdf/robotiq_85.urdf` (no ROS required).
* `load_mesh(path, mass, mesh_scale, fixed)` builds both collision and visual shapes.




## Logging (CSV Schema)

**Default columns** (subset):

* **Commanded:** `t, cmd_px, cmd_py, cmd_pz, cmd_qx, cmd_qy, cmd_qz, cmd_qw, cmd_jaw_sep`
* **Gripper actual:** `grip_px, grip_py, grip_pz, grip_qx, grip_qy, grip_qz, grip_qw, grip_jaw_sep`
* **Peg pose:** `peg_px, peg_py, peg_pz, peg_qx, peg_qy, peg_qz, peg_qw`
* **Contact sums:** `f_gripper_peg, f_peg_table, f_peg_cuboid, f_cuboid_table`

**Per‑scenario outputs**

```
logs/
 ├─ scenario1_teleop.csv
 └─ scenario2_agent.csv
```


## Tips & Tuning

**Tele‑op**

* Start with jaws open → approach peg → close to grasp (watch contact force) → lift → align → insert.
* Use `k/j` to change step size; hold **SHIFT** for fine moves.

**Agent**

* If insertion stalls or seems aggressive:

  * lower `insert_force_max`
  * increase vertical damping `b_z`
  * reduce vertical speed `vz_max`
* Clearance: adjust `hole_half_extents` to tighten/loosen fit.

---

## Development Notes

* Python: **3.9+** recommended
* Physics engine: **PyBullet**
* Default time step: **1/240 s** (configurable)
* Robotiq 2F‑85: standalone implementation; **no ROS required**

---

## License

Add your preferred license (e.g., **MIT**, **Apache‑2.0**).
