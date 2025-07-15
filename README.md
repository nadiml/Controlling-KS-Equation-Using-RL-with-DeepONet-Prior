# 🧠 Stabilizing the Kuramoto–Sivashinsky Equation Using Deep Reinforcement Learning with a DeepONet Prior

[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue.svg)](https://icml.cc/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation for our paper:

> **Stabilizing the Kuramoto–Sivashinsky Equation Using Deep Reinforcement Learning with a DeepONet Prior**  
> 📍 Accepted at the **Muslims in ML Workshop**, co-located with **ICML 2025**, Vancouver, Canada.

## 📝 Abstract

This paper presents a novel reinforcement learning framework that leverages DeepONet priors to stabilize the Kuramoto–Sivashinsky (KS) equation. DeepONet first learns a generalized control operator offline, which is refined online using Deep Deterministic Policy Gradient (DDPG) to adapt to trajectory-specific dynamics. The approach achieves a 55\% energy reduction within 0.2 time units and narrows chaotic fluctuations significantly, outperforming traditional feedback control. DeepONet reduces MSE by 99.3, while the RL agent improves mean episode reward by 59.3. The method offers a scalable and effective solution for controlling complex, high-dimensional nonlinear systems.
## 👨‍🔬 Authors

- Nadim Ahmed — [@nadiml](https://github.com/nadiml)
- Md. Ashraful Babu  
- Md. Mortuza Ahmmed  
- M. Mostafizur Rahman  
- [Mufti Mahmud](https://scholar.google.com/citations?user=L8em2YoAAAAJ&hl=en)

## 🔗 Links

- 📄 [Paper (coming soon)]()
- 🎥 [Presentation Video](https://www.youtube.com/watch?v=3eUBd3gUv88)
- 📊 Muslims in ML Workshop: [https://muslimsinml.substack.com/](https://muslimsinml.substack.com/)
- 🌐 [ICML 2025 Conference](https://icml.cc/)

## 🧰 Requirements

- Python 3.9+
- PyTorch ≥ 1.13
- NumPy, SciPy
- Matplotlib
- Gym / Stable-Baselines3 (for reinforcement learning)

Install dependencies using:

```bash
pip install -r requirements.txt
🚀 Usage
To train the DRL + DeepONet hybrid model:

bash
Copy
Edit
python train_with_deeponet_prior.py



📊 Results
We observed:

 ➢ 55% energy drop in the first 0.2 time units; 5 times overall reduction.
 ➢ DeepONet MSE ↓ 99.3%; RL mean reward ↑ 59.3%.
 ➢ RL maintains u ∈ [−8,4] vs. [−16, 12] under classic feedback

Full quantitative and visual results are included in the paper.
