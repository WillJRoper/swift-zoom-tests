import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the SWIFT directory to the path
sys.path.append(
    "/snap8/scratch/dp004/dc-rope1/SWIFT/swiftsim/tools/task_plots/"
)

from task_parser import TaskParser

# Define the base directory for the task files
base = "../../../DMO/L0100N0169NBKG0064R5p0"

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
for test in non_periodic_tests:
    runs[test] = TaskParser(f"{base}/{test}/thread_info-step64.dat")
    print(test, runs[test].get_tasks()[-1])

# Plot diagnostic histograms
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy()

for name, run in runs.items():
    labels, counts = np.unique(run.task_labels, return_counts=True)

    ax.bar(labels, counts, label=run)

ax.set_xlabel("Task")
ax.set_ylabel("Count")

ax.legend()

fig.savefig(f"plots/{test}.png")
