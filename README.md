# Bio-Inspired Swarm Navigation on Resource

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-orange.svg)](https://docs.ros.org/en/humble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19097072.svg)](https://doi.org/10.5281/zenodo.19097072)

> **Author:** Chandan Sheikder — `chandan@bit.edu.cn` — Beijing Institute of Technology (BIT)

A ROS 2 workspace implementing bio-inspired swarm navigation for the **FormicaBot** multi-robot platform. The system replicates ant colony pheromone communication and collective foraging to achieve emergent, decentralized navigation in resource-constrained environments.

> 📖 **Read the paper:** [Bio-Inspired Swarm Navigation on Resource-Constrained Robots for GPS-Denied Environments](https://www.mdpi.com/1424-8220/26/11/3525)


---

## ✨ Key Features

- **🐜 Pheromone-Based Communication** — Simulated chemical trail deposition and following for collective path formation
- **🤖 Multi-Robot Coordinator** — Decentralized swarm control without a central server
- **📊 Real-Time Analysis** — Live result processing scripts for foraging efficiency and convergence metrics
- **🔧 ROS 2 Native** — Built on `colcon` with full `ament_python` packaging

---

## 📁 Project Structure

```
Bio-Inspired-Swarm-Navigation-on-Resource/
├── formicabot_ws/
│   └── src/
│       ├── formicabot_nav/        # Core navigation stack
│       ├── formicabot_swarm/      # Swarm coordination nodes
│       └── formicabot_sim/        # Gazebo simulation world
├── process_results.py             # Post-simulation data analysis
├── ros_diag.py                    # ROS topic diagnostics helper
└── verify_logic.py                # Unit-level logic verification
```

---

## 🚀 Setup and Installation

### Prerequisites
- Ubuntu 22.04 (or Docker)
- ROS 2 Humble ([install guide](https://docs.ros.org/en/humble/Installation.html))
- Python 3.8+

### Build

```bash
git clone https://github.com/Chandan118/Bio-Inspired-Swarm-Navigation-on-Resource.git
cd Bio-Inspired-Swarm-Navigation-on-Resource/formicabot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## ▶️ Running the Simulation

```bash
# Launch swarm in Gazebo
ros2 launch formicabot_sim swarm_world.launch.py

# In a second terminal — start navigation stack
source install/setup.bash
ros2 launch formicabot_nav swarm_nav.launch.py
```

### Analyze Results

```bash
python3 process_results.py
```

---

## 📊 Diagnostics

```bash
# Check all ROS topics and node health
python3 ros_diag.py

# Verify logic consistency
python3 verify_logic.py
```

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{sheikder2025swarm,
  author    = {Chandan Sheikder},
  title     = {Bio-Inspired Swarm Navigation on Resource},
  year      = {2025},
  publisher = {GitHub},
  doi       = {10.5281/zenodo.19097072},
  url       = {https://github.com/Chandan118/Bio-Inspired-Swarm-Navigation-on-Resource}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Chandan Sheikder**  
Graduate Research Assistant, Beijing Institute of Technology  
📧 chandan@bit.edu.cn | 📞 +8618222390506  
🌐 [chandan118.github.io](https://chandan118.github.io) | [Google Scholar](https://scholar.google.com/citations?user=UWNJ6TwAAAAJ&hl=en)
