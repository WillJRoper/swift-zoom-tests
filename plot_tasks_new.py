import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the SWIFT directory to the path
sys.path.append(
    "/snap8/scratch/dp004/dc-rope1/SWIFT/swiftsim/tools/task_plots/"
)

from task_parser import TaskParser

# Define the base directory for the task files
base = "/snap8/scratch/dp004/dc-rope1/SWIFT/DMO/L0100N0169NBKG0064R5p0/"

# Define the branches
branches = ["zoom_long_range", "zoom_tl_void_mm"]

# Define the test directories
tests = [
    "adaptive_nonperiodic_tasks",
    "adaptive_periodic_tasks",
    "geometric_nonperiodic_tasks",
    "geometric_periodic_tasks",
]


# Define the non periodic tests
non_periodic_tests = [
    "adaptive_nonperiodic_tasks",
    "geometric_nonperiodic_tasks",
]

# Parse all the task files
runs = {}
for branch in branches:
    for test in non_periodic_tests:
        runs[branch + "/" + test] = TaskParser(
            f"{base}/{branch}/{test}/thread_info-step64.dat"
        )
        print(branch, test, runs[branch + "/" + test].get_tasks()[-1])

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.set_xscale("log")

for i, (name, run) in enumerate(runs.items()):
    labels, counts = np.unique(run.task_labels, return_counts=True)

    # Sort the labels and counts by counts in descending order
    sorted_indices = np.argsort(-counts)
    labels = labels[sorted_indices]
    counts = counts[sorted_indices]

    # Calculate positions for horizontal bars
    positions = np.arange(len(labels))

    # Compute the width between labels
    width = 0.8 / (len(runs) + 1)

    # Create horizontal bar plot
    ax.barh(
        positions + (i * width) + (width * 0.5),
        counts,
        height=0.75 / len(runs),
        label=name,
    )

ax.set_yticks(np.arange(len(labels)) + 0.2)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.set_xlabel("Count")
ax.set_ylabel("Task")

# Place the legend at the bottom of the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

fig.tight_layout()
fig.savefig("plots/task_count_comp_horizontal.png", bbox_inches="tight")
