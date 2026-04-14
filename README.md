# LEASH: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model

## Update

- This paper has been accepted to the Main Conference of ACL 2026 (The 64th Annual Meeting of the Association for Computational Linguistics).

LEASH (adaptive **LE**ngth pen**A**lty and reward **SH**aping) is an RL-based method that uses a Lagrangian primal-dual mechanism to dynamically control reasoning length in LLMs. It reduces average generation length by 60% while maintaining competitive accuracy.

## Setup

```bash
# Prerequisites: conda, CUDA 12.4+
bash setup.sh          # creates conda env "leash" and installs all dependencies
conda activate leash
```

## Training

1. Download model checkpoints:
   - [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
   - [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)

2. Edit `MODEL_PATH` in the training script to point to your downloaded checkpoint, then run:

```bash
# DeepSeek-R1-Distill-Qwen-1.5B, target 4k tokens
bash scripts/leash_r1_1.5b_target4k.sh

# Qwen3-4B-Thinking, target 12k tokens
bash scripts/leash_qwen3_4b_target12k.sh
```

Training data (`data/train/4k_high_quality_deepmath.parquet`, 3,939 samples) and evaluation benchmarks (`data/eval/`) are included in the repo.


## Algorithm

The core LEASH implementation is in `verl/recipe/leash/`. See [verl/recipe/leash/README.md](verl/recipe/leash/README.md) for the full algorithm description.


## Repo Structure

```
LEASH/
├── setup.sh                          # Environment setup
├── scripts/
│   ├── leash_r1_1.5b_target4k.sh     # 1.5B training script
│   └── leash_qwen3_4b_target12k.sh   # 4B training script
├── data/
│   ├── train/                         # 3,939-sample filtered DAPO-Math-17k
│   └── eval/                          # AIME24, AIME25, HMMT25, AMC, GPQA
├── verl/                              # verl framework with LEASH recipe
│   └── recipe/leash/                  # Core LEASH implementation
├── utils/reward_utils/                # Reward function
└── figures/                           # Paper figures + plotting scripts
```

## Citation

```bibtex
@article{li2025leash,
  title={Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model},
  author={Li, Yanhao and Ma, Lu and Zhang, Jiaran and Tang, Lexiang and Zhang, Wentao and Luo, Guibo},
  journal={arXiv preprint arXiv:2512.21540},
  year={2025}
}
```

## Acknowledgements

We thank [verl](https://github.com/volcengine/verl) for providing the awesome open-source RL infrastructure, and [DAPO](https://github.com/BytedTsinghua-SIA/DAPO) for the training recipe that LEASH builds upon. The model was trained on publicly available research papers and academic articles. All data used was obtained from publicly accessible sources.
