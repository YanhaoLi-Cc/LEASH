"""
Figure 1: Training dynamics of LEASH and LEASH-C under Lt = 4k on the 1.5B base.
Left: average tokens per response vs. training steps.
Right: average accuracy vs. training steps.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

df_leash = pd.read_csv(os.path.join(SCRIPT_DIR, 'data/figure1_leash.csv'))
df_const = pd.read_csv(os.path.join(SCRIPT_DIR, 'data/figure1_leash_c.csv'))

max_steps = 500

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

color_leash = '#B3D9FF'
color_const = '#FFB3D9'
lw = 1.5
ms = 8
mf = max(len(df_leash) // 12, 1)

common_kw = dict(linewidth=lw, linestyle='-', markersize=ms, markevery=mf,
                 markeredgewidth=1.2, markeredgecolor='white', alpha=0.9)

# Left: Average Tokens
ax1.plot(df_leash['_step'], df_leash['result/avg_len'], color=color_leash, marker='o',
         markerfacecolor=color_leash, label=r'LEASH-1.5B$_{L_t=\mathrm{4k}}$', **common_kw)
ax1.plot(df_const['_step'], df_const['result/avg_len'], color=color_const, marker='s',
         markerfacecolor=color_const, label=r'LEASH-C-1.5B$_{L_t=\mathrm{4k}}$', **common_kw)
ax1.set(xlabel='Training Steps', ylabel='Average Tokens', title='Average Tokens During Training',
        xlim=(-20, max_steps + 20))

# Right: Average Accuracy
ax2.plot(df_leash['_step'], df_leash['result/avg_acc'], color=color_leash, marker='o',
         markerfacecolor=color_leash, label=r'LEASH-1.5B$_{L_t=\mathrm{4k}}$', **common_kw)
ax2.plot(df_const['_step'], df_const['result/avg_acc'], color=color_const, marker='s',
         markerfacecolor=color_const, label=r'LEASH-C-1.5B$_{L_t=\mathrm{4k}}$', **common_kw)
ax2.set(xlabel='Training Steps', ylabel='Average Accuracy', title='Accuracy During Training',
        ylim=(0.12, 0.28), xlim=(-20, max_steps + 20))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

for ax in [ax1, ax2]:
    ax.legend(fontsize=13, frameon=True, edgecolor='black', framealpha=1, borderpad=0.8)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=13, width=1.0, length=5)
    ax.set_facecolor('#FAFAFA')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

fig.patch.set_facecolor('white')
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(SCRIPT_DIR, 'figure1.pdf'), format='pdf', dpi=500, bbox_inches='tight')
print("Saved figure1.pdf")
plt.show()
