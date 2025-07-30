import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from config import settings
from matplotlib.colors import LinearSegmentedColormap


def get_user_complete_data(uid, parquet_file):
    """Get all data for a specific user"""
    filters = [("uid", "=", uid)]
    table = pq.read_table(parquet_file, filters=filters)
    return table.to_pandas()


def plot_user_with_deltas(uid, parquet_file, save_dir="./"):
    """
    Plot user trajectory with delta plots to the right
    Layout: [Trajectory] [Delta X]
                        [Delta Y]
    """
    print(f"Plotting trajectory and deltas for user {uid}...")

    # Get user data
    user_data = get_user_complete_data(uid, parquet_file)

    if len(user_data) == 0:
        print(f"No data found for user {uid}")
        return

    # Sort by timestamp and add day/time columns
    user_data = user_data.sort_values("ts")
    user_data["day"] = user_data["ts"] // 48 + 1

    # Use ALL data (train + test)
    all_data = user_data.copy()

    print(f"User {uid}: {len(all_data)} total points")

    # Calculate deltas for all data
    all_data["dx"] = all_data["x"].diff()
    all_data["dy"] = all_data["y"].diff()
    delta_data = all_data.dropna()  # Remove first row with NaN

    # Separate train and test for trajectory coloring
    train_data = all_data[all_data["day"] <= settings.TRAIN_DAYS]
    test_data = all_data[all_data["day"] > settings.TRAIN_DAYS]

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 8))

    # Create grid layout: trajectory on left, deltas stacked on right
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    ax_traj = fig.add_subplot(gs[:, 0])  # Trajectory spans both rows on left
    ax_dx = fig.add_subplot(gs[0, 1])  # Delta X on top right
    ax_dy = fig.add_subplot(gs[1, 1])  # Delta Y on bottom right

    # Time index for delta plots
    time_index = range(len(delta_data))

    # Plot Delta X
    ax_dx.plot(time_index, delta_data["dx"], "b-", alpha=0.7, linewidth=1)
    ax_dx.scatter(time_index, delta_data["dx"], s=8, alpha=0.6, c="blue")
    ax_dx.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax_dx.set_xlabel("Time Step")
    ax_dx.set_ylabel("Delta X")
    ax_dx.set_title(f"ΔX over Time")
    ax_dx.grid(True, alpha=0.3)

    # Plot Delta Y
    ax_dy.plot(time_index, delta_data["dy"], "g-", alpha=0.7, linewidth=1)
    ax_dy.scatter(time_index, delta_data["dy"], s=8, alpha=0.6, c="green")
    ax_dy.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax_dy.set_xlabel("Time Step")
    ax_dy.set_ylabel("Delta Y")
    ax_dy.set_title(f"ΔY over Time")
    ax_dy.grid(True, alpha=0.3)

    # Plot Trajectory with temperature colormap
    if len(all_data) > 1:
        # Plot training data in blue
        if len(train_data) > 0:
            ax_traj.plot(
                train_data["x"],
                train_data["y"],
                color="blue",
                alpha=0.6,
                linewidth=1,
                label="Training",
            )
            ax_traj.scatter(
                train_data["x"].iloc[0],
                train_data["y"].iloc[0],
                color="green",
                s=100,
                marker="o",
                label="Train Start",
                zorder=10,
                edgecolor="black",
            )

        # Plot test data with temperature colormap
        if len(test_data) > 0:
            # Create temperature colormap for test data (blue to red)
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(test_data)))

            # Plot line segments with changing color
            for i in range(len(test_data) - 1):
                ax_traj.plot(
                    [test_data["x"].iloc[i], test_data["x"].iloc[i + 1]],
                    [test_data["y"].iloc[i], test_data["y"].iloc[i + 1]],
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                )

            # Plot points with temperature colors
            scatter = ax_traj.scatter(
                test_data["x"],
                test_data["y"],
                c=range(len(test_data)),
                cmap="coolwarm",
                s=30,
                alpha=0.8,
                zorder=5,
            )

            # Mark test start and end
            ax_traj.scatter(
                test_data["x"].iloc[0],
                test_data["y"].iloc[0],
                color="orange",
                s=100,
                marker="s",
                label="Test Start",
                zorder=10,
                edgecolor="black",
            )
            ax_traj.scatter(
                test_data["x"].iloc[-1],
                test_data["y"].iloc[-1],
                color="red",
                s=100,
                marker="s",
                label="Test End",
                zorder=10,
                edgecolor="black",
            )

            # Add colorbar for test data
            cbar = plt.colorbar(scatter, ax=ax_traj)
            cbar.set_label("Test Time Progress")

    ax_traj.set_xlim(1, 200)
    ax_traj.set_ylim(1, 200)
    ax_traj.set_xlabel("X Coordinate")
    ax_traj.set_ylabel("Y Coordinate")
    ax_traj.set_title(f"User {uid} Complete Trajectory")
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)

    # Add statistics
    if len(delta_data) > 0:
        dx_nonzero = delta_data["dx"][delta_data["dx"] != 0]
        dy_nonzero = delta_data["dy"][delta_data["dy"] != 0]

        dx_stats = f"Non-zero: {len(dx_nonzero)}/{len(delta_data)} ({len(dx_nonzero)/len(delta_data)*100:.1f}%)"
        dy_stats = f"Non-zero: {len(dy_nonzero)}/{len(delta_data)} ({len(dy_nonzero)/len(delta_data)*100:.1f}%)"

        if len(dx_nonzero) > 0:
            dx_stats += f"\nRange: [{dx_nonzero.min()}, {dx_nonzero.max()}]"
        if len(dy_nonzero) > 0:
            dy_stats += f"\nRange: [{dy_nonzero.min()}, {dy_nonzero.max()}]"

        ax_dx.text(
            0.02,
            0.98,
            dx_stats,
            transform=ax_dx.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )
        ax_dy.text(
            0.02,
            0.98,
            dy_stats,
            transform=ax_dy.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )

        # Trajectory statistics
        train_unique = (
            len(train_data[["x", "y"]].drop_duplicates()) if len(train_data) > 0 else 0
        )
        test_unique = (
            len(test_data[["x", "y"]].drop_duplicates()) if len(test_data) > 0 else 0
        )

        traj_stats = f"Train: {len(train_data)} pts, {train_unique} unique\n"
        traj_stats += f"Test: {len(test_data)} pts, {test_unique} unique"

        ax_traj.text(
            0.02,
            0.98,
            traj_stats,
            transform=ax_traj.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    filename = f"{save_dir}user_{uid}_complete_trajectory.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


def main():
    parquet_file = "city_B_challengedata_converted.parquet"
    uids_to_plot = [11764, 24271, 27001]

    print("Creating trajectory plots with deltas...")

    # Plot each user
    for uid in uids_to_plot:
        plot_user_with_deltas(uid, parquet_file)

    print("All plots created successfully!")


if __name__ == "__main__":
    main()
