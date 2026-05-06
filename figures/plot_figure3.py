"""
Figure 3: Evolution of the model's thinking patterns across RL iterations
on AIME2024 and AIME2025, using DeepSeek-R1-Distill-Qwen-1.5B as the base model.

Data pre-computed from evaluation outputs at each checkpoint by counting
keyword frequencies (summary, rethink, plan) per sample.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# Pre-computed from evaluation outputs (480 samples per step)
step = [0, 80, 160, 240, 320, 400]
avg_tokens = [16886, 14079, 9725, 7248, 7021, 6193]
summary_num = [63.54, 53.2, 35.1, 23.66, 25.27, 25.98]
rethink_num = [9.38, 7.71, 5.06, 3.62, 3.68, 3.69]
plan_num = [30.53, 19.41, 14.89, 12.17, 9.74, 9.16]

fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=500)

behaviors = [
    (avg_tokens, 'Average Tokens', '#E67E22', 'v', 'Token Count'),
    (summary_num, 'Summary Behavior', '#FF6B6B', 'o', 'Frequency'),
    (rethink_num, 'Rethink Behavior', '#4ECDC4', 's', 'Frequency'),
    (plan_num, 'Plan Behavior', '#45B7D1', '^', 'Frequency'),
]

for ax, (data, title, color, marker, ylabel) in zip(axes, behaviors):
    ax.plot(step, data, marker=marker, markersize=8, linewidth=3,
            color=color, alpha=0.8, markeredgecolor='black', markeredgewidth=1)

    ax.set(xlabel='Training Steps', ylabel=ylabel, title=title,
           xlim=(-40, 440))
    ax.set_facecolor('#FAFAFA')

    if 'Tokens' in title:
        ax.set_ylim(0, max(data) * 1.1)
    else:
        pad = (max(data) - min(data)) * 0.15
        ax.set_ylim(max(0, min(data) - pad), max(data) + pad)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
        spine.set_visible(True)

    ax.grid(True, alpha=0.5, linestyle='--', linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=18, colors='#000000')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(20)

    for x, y in zip(step, data):
        label = f'{y/1000:.1f}K' if 'Tokens' in title and y >= 1000 else f'{y}'
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=16, color='#000000')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'figure3.pdf'), format='pdf', dpi=500, bbox_inches='tight')
print("Saved figure3.pdf")
plt.show()
