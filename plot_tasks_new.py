import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the SWIFT directory to the path
sys.path.append(
    "/snap8/scratch/dp004/dc-rope1/SWIFT/swiftsim/tools/task_plots/"
)

from task_parser import TaskParser


def make_mask(
    run,
    ci_type=None,
    cj_type=None,
    ci_subtype=None,
    cj_subtype=None,
    depth=None,
):
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

    return mask


def make_task_hist_split(runs):
    fig = plt.figure(figsize=(12, 16))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.grid(True)

    # Combine all information into the labels
    labels_dict = {
        name: np.zeros(run.ntasks, dtype=object) for name, run in runs.items()
    }
    types = {}
    for name, run in runs.items():
        for i in range(run.ntasks):
            task = run.task_labels[i]
            types[task] = types.get(task, 0) + 1
            if "pair" in task:
                labels_dict[name][i] = (
                    f"{task}:"
                    f"{run.tasks[i].ci_type}({run.tasks[i].ci_subtype})"
                    f"@{run.tasks[i].ci_depth}->"
                    f"{run.tasks[i].ci_type}({run.tasks[i].ci_subtype})"
                    f"@{run.tasks[i].cj_depth}"
                )
            else:
                labels_dict[name][i] = (
                    f"{task}:"
                    f"{run.tasks[i].ci_type}({run.tasks[i].ci_subtype})"
                    f"@{run.tasks[i].ci_depth}"
                )

    # Get the sorting indices based on types
    types_arr = list(types.keys())
    counts_arr = np.array(list(types.values()))
    sinds_arr = np.argsort(-counts_arr)
    sinds = {k: ind for k, ind in zip(types_arr, sinds_arr)}

    for i, (name, run) in enumerate(runs.items()):
        labels, counts = np.unique(labels_dict[name], return_counts=True)

        # Sort the labels and counts by counts in descending order
        sorted_indices = np.array([sinds[k.split(":")[0]] for k in labels])
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

    ax.set_yticks(np.arange(len(labels)) + 0.2)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Count")

    # Place the legend at the bottom of the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Define the filename
    filename = "plots/task_count_comp_split.png"

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")

    plt.close(fig)


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
        mask = make_mask(run, ci_type, cj_type, ci_subtype, cj_subtype, depth)

        labels, counts = np.unique(run.task_labels[mask], return_counts=True)

        # Sort the labels and counts by counts in descending order
        if i == 0:
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

    ax.set_yticks(np.arange(len(labels)) + 0.2)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Count")

    # Place the legend at the bottom of the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Define the filename
    filename = "plots/task_count_comp"
    if ci_type is not None and cj_type is not None:
        filename += f"_types{ci_type}-{cj_type}"
    if ci_subtype is not None and cj_subtype is not None:
        filename += f"_subtypes{ci_subtype}-{cj_subtype}"
    if ci_type is not None and cj_type is None:
        filename += f"_type{ci_type}"
    if ci_subtype is not None and cj_subtype is None:
        filename += f"_subtype{ci_subtype}"
    if ci_type is None and cj_type is not None:
        filename += f"_type{cj_type}"
    if ci_subtype is None and cj_subtype is not None:
        filename += f"_subtype{cj_subtype}"
    if depth is not None:
        filename += f"_depth{depth}"

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")

    plt.close(fig)


def make_task_hist_time_weighted(
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
        mask = make_mask(run, ci_type, cj_type, ci_subtype, cj_subtype, depth)

        # Loop over tasks collecting their runtime
        labels = np.unique(run.task_labels[mask])
        counts = np.array(
            [np.sum(run.dt[mask][run.task_labels[mask] == k]) for k in labels]
        )

        # Sort the labels and counts by counts in descending order
        if i == 0:
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

    ax.set_yticks(np.arange(len(labels)) + 0.2)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Time (ms)")

    # Place the legend at the bottom of the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Define the filename
    filename = "plots/task_time_comp"
    if ci_type is not None and cj_type is not None:
        filename += f"_types{ci_type}-{cj_type}"
    if ci_subtype is not None and cj_subtype is not None:
        filename += f"_subtypes{ci_subtype}-{cj_subtype}"
    if ci_type is not None and cj_type is None:
        filename += f"_type{ci_type}"
    if ci_subtype is not None and cj_subtype is None:
        filename += f"_subtype{ci_subtype}"
    if ci_type is None and cj_type is not None:
        filename += f"_type{cj_type}"
    if ci_subtype is None and cj_subtype is not None:
        filename += f"_subtype{cj_subtype}"
    if depth is not None:
        filename += f"_depth{depth}"

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")

    plt.close(fig)


def make_pair_mindist_plot(
    runs,
    ci_type=None,
    cj_type=None,
    ci_subtype=None,
    cj_subtype=None,
    depth=None,
    nbins=30,
):
    # Make the figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_yscale("log")
    ax.grid(True)

    # Collect the distances
    dists = {}
    for i, (name, run) in enumerate(runs.items()):
        mask = make_mask(run, ci_type, cj_type, ci_subtype, cj_subtype, depth)

        # Ensure we only have pair tasks (i.e. the string "pair" is in the
        # task label)
        mask = np.logical_and(
            mask, np.array(["pair" in t for t in run.task_labels])
        )

        # Get the distances
        dists[name] = run.min_dists[mask]

    # Construct the bins
    all_dists = np.concatenate(list(dists.values()))
    bins = np.linspace(all_dists.min(), all_dists.max(), nbins + 1)
    bin_cents = (bins[:-1] + bins[1:]) / 2

    # Compute histogram and plot
    for name in dists.keys():
        linestyle = "--" if "long_range" in name else "-"
        H, _ = np.histogram(dists[name], bins=bins)
        ax.plot(bin_cents, H, label=name, linestyle=linestyle)

    ax.set_xlabel("sqrt(cell_min_dist2) (U_L)")
    ax.set_ylabel("Count")

    # Place the legend at the bottom of the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Define the filename
    filename = "plots/pair_min_dist_comp"
    if ci_type is not None and cj_type is not None:
        filename += f"_types{ci_type}-{cj_type}"
    if ci_subtype is not None and cj_subtype is not None:
        filename += f"_subtypes{ci_subtype}-{cj_subtype}"
    if ci_type is not None and cj_type is None:
        filename += f"_type{ci_type}"
    if ci_subtype is not None and cj_subtype is None:
        filename += f"_subtype{ci_subtype}"
    if ci_type is None and cj_type is not None:
        filename += f"_type{cj_type}"
    if ci_subtype is None and cj_subtype is not None:
        filename += f"_subtype{cj_subtype}"
    if depth is not None:
        filename += f"_depth{depth}"

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")

    plt.close(fig)


def make_pair_mpoledist_plot(
    runs,
    ci_type=None,
    cj_type=None,
    ci_subtype=None,
    cj_subtype=None,
    depth=None,
    nbins=30,
):
    # Make the figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_yscale("log")
    ax.grid(True)

    # Collect the distances
    dists = {}
    for i, (name, run) in enumerate(runs.items()):
        mask = make_mask(run, ci_type, cj_type, ci_subtype, cj_subtype, depth)

        # Ensure we only have pair tasks (i.e. the string "pair" is in the
        # task label)
        mask = np.logical_and(
            mask, np.array(["pair" in t for t in run.task_labels])
        )

        # Get the distances
        dists[name] = run.mpole_dists[mask]

    # Construct the bins
    all_dists = np.concatenate(list(dists.values()))
    bins = np.linspace(all_dists.min(), all_dists.max(), nbins + 1)
    bin_cents = (bins[:-1] + bins[1:]) / 2

    # Compute histogram and plot
    for name in dists.keys():
        linestyle = "--" if "long_range" in name else "-"
        H, _ = np.histogram(dists[name], bins=bins)
        ax.plot(bin_cents, H, label=name, linestyle=linestyle)

    ax.set_xlabel("Multipole CoM distance (U_L)")
    ax.set_ylabel("Count")

    # Place the legend at the bottom of the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Define the filename
    filename = "plots/pair_mpole_dist_comp"
    if ci_type is not None and cj_type is not None:
        filename += f"_types{ci_type}-{cj_type}"
    if ci_subtype is not None and cj_subtype is not None:
        filename += f"_subtypes{ci_subtype}-{cj_subtype}"
    if ci_type is not None and cj_type is None:
        filename += f"_type{ci_type}"
    if ci_subtype is not None and cj_subtype is None:
        filename += f"_subtype{ci_subtype}"
    if ci_type is None and cj_type is not None:
        filename += f"_type{cj_type}"
    if ci_subtype is None and cj_subtype is not None:
        filename += f"_subtype{cj_subtype}"
    if depth is not None:
        filename += f"_depth{depth}"

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Define the base directory for the task files
    base = "/snap8/scratch/dp004/dc-rope1/SWIFT/DMO/L0100N0169NBKG0064R5p0/"

    # Define the branches
    branches = [
        "zoom_tl_void_mm",
        "zoom_neighbour_splitting",
    ]

    # Define the test directories
    tests = [
        # "adaptive_nonperiodic_tasks",
        "adaptive_periodic256_tasks",
        "adaptive_periodic512_tasks",
        # "geometric_nonperiodic_tasks",
        # "geometric_periodic_tasks",
    ]

    # Define the non periodic tests
    non_periodic_tests = [
        "adaptive_nonperiodic_tasks",
        "geometric_nonperiodic_tasks",
    ]

    # Parse all the task files
    runs = {}
    for branch in branches:
        for test in tests:
            runs[branch + "/" + test] = TaskParser(
                f"{base}/{branch}/{test}/thread_info-step64.dat"
            )
            print(branch, test, runs[branch + "/" + test].get_tasks()[-1])

    make_task_hist_split(runs)

    make_task_hist(runs)
    make_task_hist(runs, ci_type=1, cj_type=1)
    make_task_hist(runs, ci_type=3, cj_type=3)
    make_task_hist(runs, ci_type=1, cj_type=3)

    make_task_hist(runs, depth=0)
    make_task_hist(runs, ci_type=1, cj_type=1, depth=0)
    make_task_hist(runs, ci_type=1, cj_type=3, depth=0)
    make_task_hist(runs, ci_type=3, cj_type=3, depth=0)

    make_task_hist_time_weighted(runs)
    make_task_hist_time_weighted(runs, ci_type=1, cj_type=1)
    make_task_hist_time_weighted(runs, ci_type=3, cj_type=3)
    make_task_hist_time_weighted(runs, ci_type=1, cj_type=3)

    make_task_hist_time_weighted(runs, depth=0)
    make_task_hist_time_weighted(runs, ci_type=1, cj_type=1, depth=0)
    make_task_hist_time_weighted(runs, ci_type=1, cj_type=3, depth=0)
    make_task_hist_time_weighted(runs, ci_type=3, cj_type=3, depth=0)

    make_pair_mindist_plot(runs)
    make_pair_mindist_plot(runs, ci_type=1, cj_type=1)
    make_pair_mindist_plot(runs, ci_type=3, cj_type=3)
    make_pair_mindist_plot(runs, ci_type=1, cj_type=3)

    make_pair_mpoledist_plot(runs)
    make_pair_mpoledist_plot(runs, ci_type=1, cj_type=1)
    make_pair_mpoledist_plot(runs, ci_type=3, cj_type=3)
    make_pair_mpoledist_plot(runs, ci_type=1, cj_type=3)
