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

# Plot diagnostic histograms
fig = plt.figure(figsize=(6, 12))
ax = fig.add_subplot(111)
ax.semilogy()

for name, run in runs.items():
    labels, counts = np.unique(run.task_labels, return_counts=True)
    if "long_range" in name:
        ax.plot(labels, counts, label=name)
    else:
        ax.plot(labels, counts, label=name, linestyle="--")

ax.set_xlabel("Task")
ax.set_ylabel("Count")

ax.tick_params(axis="x", labelrotation=90)

ax.legend()

fig.savefig("plots/task_count_comp.png", bbox_inches="tight")
