"""Generate report-ready figures from the deployed Streamlit load-test CSV.
Run from project root: python generate_deployed_performance_figures.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'results/prototype_testing/deployed_prediction_summary.csv'
if not CSV.exists():
    CSV = ROOT / 'deployed_prediction_summary.csv'
if not CSV.exists():
    raise FileNotFoundError('Place deployed_prediction_summary.csv in results/prototype_testing/ or beside this script.')
OUT = CSV.parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CSV).sort_values('concurrent_users')
users = df['concurrent_users'].to_numpy()
labels = [f'{u} user' if u == 1 else f'{u} users' for u in users]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(users))
w = 0.34
avg = df['average_prediction_seconds'].to_numpy()
mx = df['maximum_prediction_seconds'].to_numpy()
bars1 = ax.bar(x - w/2, avg, w, label='Average latency')
bars2 = ax.bar(x + w/2, mx, w, label='Maximum latency')
for bars in (bars1, bars2):
    ax.bar_label(bars, fmt='%.2f s', padding=3, fontsize=10)
ax.set_xticks(x, labels)
ax.set_ylim(0, max(mx) * 1.20)
ax.set_ylabel('Prediction latency (seconds)')
ax.set_xlabel('Concurrent browser sessions')
ax.set_title('Deployed Streamlit prediction latency')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=.2)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'deployed_prediction_latency.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
throughput = df['throughput_predictions_per_second'].to_numpy()
bars = ax.bar(x, throughput, width=0.52)
ax.bar_label(bars, fmt='%.3f', padding=4)
ax.set_xticks(x, labels)
ax.set_ylim(0, max(throughput) * 1.23)
ax.set_ylabel('Successful predictions per second')
ax.set_xlabel('Concurrent browser sessions')
ax.set_title('Deployed Streamlit prediction throughput')
ax.grid(axis='y', alpha=.2)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'deployed_prediction_throughput.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Created figures in: {OUT}')
