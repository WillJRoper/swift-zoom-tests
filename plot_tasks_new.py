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


def make_task_hist(
    runs,
    ci_type=None,
    cj_type=None,
    ci_subtype=None,
    cj_subtype=None,
    depth=None,
):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.grid(True)

    for i, (name, run) in enumerate(runs.items()):
        mask = np.ones(len(run.task_labels), dtype=bool)
        if (ci_type is not None and cj_type is None) or (
            ci_type is None and cj_type is not None
        ):
            cell_type = ci_type if ci_type is not None else cj_type
            mask = np.logical_and(
                mask,
                np.logical_or(
                    run.ci_types == cell_type,
                    run.cj_types == cell_type,
                ),
            )
        if ci_type is not None and cj_type is not None:
            mask = np.logical_and(
                mask,
                np.logical_or(
                    np.logical_and(
                        run.ci_types == ci_type,
                        run.cj_types == cj_type,
                    ),
                    np.logical_and(
                        run.ci_types == cj_type,
                        run.cj_types == ci_type,
                    ),
                ),
            )
        if (ci_subtype is not None and cj_subtype is None) or (
            ci_subtype is None and cj_subtype is not None
        ):
            cell_subtype = ci_subtype if ci_subtype is not None else cj_subtype
            mask = np.logical_and(
                mask,
                np.logical_or(
                    run.ci_subtypes == cell_subtype,
                    run.cj_subtypes == cell_subtype,
                ),
            )
        if ci_subtype is not None and cj_subtype is not None:
            mask = np.logical_and(
                mask,
                np.logical_or(
                    np.logical_and(
                        run.ci_subtypes == ci_subtype,
                        run.cj_subtypes == cj_subtype,
                    ),
                    np.logical_and(
                        run.ci_subtypes == cj_subtype,
                        run.cj_subtypes == ci_subtype,
                    ),
                ),
            )
        if depth is not None:
            mask = np.logical_and(
                mask,
                np.logical_or(run.ci_depths == depth, run.cj_depths == depth),
            )

        labels, counts = np.unique(run.task_labels[mask], return_counts=True)

        # Sort the labels and counts by counts in descending order
        sorted_indices = np.argsort(-counts)
        labels = labels[sorted_indices]
        counts = counts[sorted_indices]

        # Calculate positions for horizontal bars
        positions = np.arange(len(labels))

        # Compute the width between labels
        width = 0.8 / (len(runs) + 1)

        # Create horizontal bar plot
        bars = ax.barh(
            positions + (i * width),
            counts,
            height=0.75 / len(runs),
            label=name,
            alpha=0.7,
        )

        if "long_range" in name:
            # Adding hatching
            for bar in bars:
                bar.set_hatch("//")
                bar.set_edgecolor(bar.get_facecolor())

    ax.set_yticks(np.arange(len(labels)) + 0.2)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Count")

    # Place the legend at the bottom of the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Define the filename
    filename = "plots/task_count_comp"
    if cell_type is not None:
        filename += f"_type{cell_type}"
    if cell_subtype is not None:
        filename += f"_subtype{cell_subtype}"
    if depth is not None:
        filename += f"_depth{depth}"

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")


make_task_hist(runs)
make_task_hist(runs, ci_type=1, cj_type=1)
make_task_hist(runs, ci_type=3, cj_type=3)
make_task_hist(runs, ci_type=1, cj_type=3)
