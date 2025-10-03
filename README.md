# Peg Insertion Simulation (PyBullet + Robotiq 2F-85)

This project simulates a **peg-insertion task** using the Robotiq 2F-85 gripper and PyBullet physics engine.
It supports two control modes:

1. **Tele-Operated Control (Scenario 1)** — move the gripper in real-time using keyboard inputs.
2. **Simple Agent (Scenario 2)** — a scripted agent that automatically picks up and inserts the peg.

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/peg-insertion-sim.git
cd peg-insertion-sim
```

### 2. Create and activate a Python virtual environment

```bash
# Create venv
python3 -m venv bullet_env

# Activate venv
# On Linux / macOS:
source bullet_env/bin/activate

# On Windows (PowerShell):
.\bullet_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -e .
```

---

## ▶️ Running the Simulation

### Scenario 1: Tele-Operated Control

Launch simulation with **keyboard control**:

```bash
python run_scenario1_teleop.py
```

**Controls:**

* **Translation:**

  * `s/x`: forward/back
  * `z/x`: left/right
  * `a/d`: up/down

* **Rotation:**

  * Arrow keys: yaw/pitch
  * `,` `.`: roll

* **Gripper:**

  * `[` close jaws
  * `]` open jaws

**Goal:** Move above the peg, close the gripper to grasp, lift, align with the cuboid hole, and insert.

---

### Scenario 2: Simple Agent

Run with a **scripted autonomous agent**:

```bash
python run_scenario2_agent.py
```

The agent will:

1. Move above the peg.
2. Descend and close gripper until grasp is confirmed.
3. Lift peg (checking for slip).
4. Move above the cuboid hole.
5. Insert peg.

---

## 📂 Project Structure

```
peg-insertion-sim/
│
├── env_peg_insertion.py   # Environment (physics, logging, success check)
├── controllers.py         # Teleop + SimpleAgent controllers
├── run_scenario1_teleop.py # Tele-operated demo
├── run_scenario2_agent.py  # Agent demo
├── assets/                # STL meshes (peg.stl, cuboid.stl) + URDF (robotiq_85.urdf)
├── logs/                  # CSV logs of simulation runs
└── README.md
```

---

## 📊 Logs

Each run saves a CSV log under `logs/` with:

* Commanded gripper pose and jaw separation
* Actual gripper pose and jaw separation
* Peg position and orientation
* Contact forces (gripper–peg, peg–table, peg–cuboid, cuboid–table)

You can analyze these logs with Python, pandas, or plot them with matplotlib.

---

## ✅ Notes

* The simulation camera defaults to a **close-up view of the peg & cuboid**.
* Adjust `EnvConfig` parameters in `env_peg_insertion.py` to tweak time step, solver iterations, or jaw mapping.
* For smooth performance, run with **PyBullet GUI** (`p.GUI`).
* For headless training, use `p.DIRECT`.

---

## 🚀 Next Steps

* Add a Gymnasium wrapper for RL training.
* Improve success criteria (tight clearance checking).
* Extend to multi-peg or randomized hole locations.
