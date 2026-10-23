from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Run this script from your project's root folder.
INPUT = Path('results/prototype_testing/performance_summary.csv')
OUTPUT = Path('results/prototype_testing')
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
rows = df[df['measurement_location'] == 'Local Python model inference'].copy()
rows = rows.sort_values('concurrent_users')
if len(rows) != 3:
    raise ValueError('Expected 3 local inference scenarios (1, 5, 10).')

labels = [f'{int(n)} concurrent call' + ('s' if int(n) != 1 else '') for n in rows['concurrent_users']]
avg_ms = rows['average_seconds'].to_numpy() * 1000
max_ms = rows['max_seconds'].to_numpy() * 1000

fig, ax = plt.subplots(figsize=(8.2, 4.8))
bars = ax.bar(labels, avg_ms, color='#315C85', width=0.55)
for bar, avg, maxval in zip(bars, avg_ms, max_ms):
    ax.annotate(f'{avg:.1f} ms', (bar.get_x() + bar.get_width()/2, avg),
                xytext=(0, 6), textcoords='offset points', ha='center', fontsize=11)

ax.set_ylabel('Average inference response time (ms)', fontsize=11)
ax.set_xlabel('Concurrent local inference calls', fontsize=11)
ax.set_title('Local BERT Inference Performance Under Concurrent Load', fontsize=13, pad=15)
ax.set_ylim(0, max(avg_ms) * 1.25)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.2)
ax.set_axisbelow(True)
fig.tight_layout()
for extension in ('png', 'pdf'):
    path = OUTPUT / f'local_inference_performance.{extension}'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
plt.show()
