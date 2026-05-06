"""
Figure 2: Training dynamics of LEASH and LEASH-C under Lt = 4k.
Plots: (1) satisfaction rate, (2) adaptive penalty coefficient lambda,
       (3) effective penalty value, (4) average token length.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

df_leash = pd.read_csv(os.path.join(SCRIPT_DIR, 'data/figure2_leash.csv'))
df_const = pd.read_csv(os.path.join(SCRIPT_DIR, 'data/figure2_leash_c.csv'))

max_steps = 500
metrics = ['constraint/satisfaction_rate', 'constraint/lambda', 'constraint/avg_penalty', 'constraint/avg_length']
titles = ['Train Set Satisfaction Rate', r'$\lambda$', 'Penalty', 'Tokens (Training)']
ylabels = ['Satisfaction Rate', r'$\lambda$', 'Penalty', 'Tokens']

fig, axes = plt.subplots(1, 4, figsize=(24, 5), dpi=500)

color_leash = '#B3D9FF'
color_const = '#FFB3D9'
lw = 1.5
ms = 8
mf1 = max(len(df_leash) // 12, 1)
mf2 = max(len(df_const) // 12, 1)

for i, (ax, metric, title, ylabel) in enumerate(zip(axes, metrics, titles, ylabels)):
    kw = dict(linewidth=lw, linestyle='-', markersize=ms, markeredgewidth=1.2,
              markeredgecolor='white', alpha=0.9)
    ax.plot(df_leash['_step'], df_leash[metric], color=color_leash, marker='o', markevery=mf1,
            markerfacecolor=color_leash, label=r'LEASH-1.5B$_{L_t=\mathrm{4k}}$', **kw)
    ax.plot(df_const['_step'], df_const[metric], color=color_const, marker='s', markevery=mf2,
            markerfacecolor=color_const, label=r'LEASH-C-1.5B$_{L_t=\mathrm{4k}}$', **kw)

    ax.set(xlabel='Training Steps', ylabel=ylabel, title=title, xlim=(-20, max_steps + 20))
    if i == 0:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    ax.legend(fontsize=11, frameon=True, edgecolor='black', framealpha=1, borderpad=0.8)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11, width=1.0, length=5)
    ax.set_facecolor('#FAFAFA')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

fig.patch.set_facecolor('white')
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(SCRIPT_DIR, 'figure2.pdf'), format='pdf', dpi=500, bbox_inches='tight')
print("Saved figure2.pdf")
plt.show()
