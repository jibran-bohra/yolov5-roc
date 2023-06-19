import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_roc_comparison(
    dir_list,
    runs_dir,
    save_dir,
    name="ROC_comparison.png",
):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Receiver Operator Characteristic Curve")

    for dir in dir_list:
        load_dir = runs_dir / Path(dir)
        AUC = np.load(Path(load_dir) / "AUC.npy")
        ROC = np.load(Path(load_dir) / "ROC_curve.npy")
        px = np.linspace(0, 1, 1000)
        py = np.stack(ROC, axis=1)
        ax.plot(
            px, py.mean(1), linewidth=3, label=f"{dir}, AUC: {AUC[:, 0].mean():.3f}"
        )

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir / name, dpi=250)
    plt.close(fig)


def plot_pr_comparison(
    dir_list,
    runs_dir,
    save_dir,
    name="PR_comparison.png",
):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-Recall Curve")

    for dir in dir_list:
        load_dir = runs_dir / Path(dir)
        AP = np.load(Path(load_dir) / "AP.npy")
        PR = np.load(Path(load_dir) / "PR_curve.npy")
        px = np.linspace(0, 1, 1000)
        py = np.stack(PR, axis=1)
        ax.plot(px, py.mean(1), linewidth=3, label=f"{dir}, AUC: {AP[:, 0].mean():.3f}")

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir / name, dpi=250)
    plt.close(fig)


def main():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    parser = argparse.ArgumentParser(
        description="Process PR and ROC comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python comparative_plots.py 'val/exp1' 'val/exp2 'val/exp3' --runs_dir 'runs' --save_dir 'runs/comparative_plots' --namePR 'my_PR_plot.png' --nameROC 'my_ROC_plot.png'
""",
    )
    parser.add_argument(
        "directories", 
        metavar="INPUT DIRECTORIES", 
        nargs="+", 
        type=str, 
        help="input directories"
    )
    parser.add_argument(
        "--runs_dir", 
        type=str, 
        default = ROOT / os.path.join(*"runs".split("/")), 
        help="directory for runs"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default = ROOT / os.path.join(*"runs/comparative".split("/")),
        help="directory for saving plots",
    )
    parser.add_argument(
        "--namePR",
        type=str,
        default="PR_comparison.png",
        help="name of the PR plot file",
    )
    parser.add_argument(
        "--nameROC",
        type=str,
        default="ROC_comparison.png",
        help="name of the ROC plot file",
    )

    args = parser.parse_args()

    # Generate a list of directories
    directory_list = args.directories
    runs_dir = Path(args.runs_dir)
    save_dir = Path(args.save_dir)

    # Print the list of directories
    print("Input Directories:", directory_list)             #These are folders within the runs directory
    print("ROOT:", ROOT)                                    #Root directory; all paths are relative to this directory
    print("runs_dir: ", runs_dir)                           #runs directory; input directories must be contained here
    print("save_dir:", save_dir)                            #save directory; plots will be output here. 

    plot_pr_comparison(
        directory_list,
        runs_dir= Path(args.runs_dir),
        save_dir= Path(args.save_dir),
        name=args.namePR,
    )
    
    plot_roc_comparison(
        directory_list,
        runs_dir= Path(args.runs_dir),
        save_dir= Path(args.save_dir),
        name=args.nameROC,
    )


if __name__ == "__main__":
    main()
