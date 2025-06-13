import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import seaborn as sns


def visualize_mobility_paths(csv_file, num_users=100, save_plots=True):
    """
    Visualize mobility paths of top users on 200x200 grid
    """

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    print(f"Loaded data shape: {df.shape}")
    print(f"Total users available: {len(df)}")

    # Take first num_users (they should be sorted by data density)
    users_to_plot = df.head(num_users)
    user_ids = users_to_plot["uid"].tolist()

    print(f"Visualizing first {len(user_ids)} users: {user_ids[:10]}...")

    # Convert wide format back to trajectory data
    trajectories = defaultdict(list)

    print("Converting wide format to trajectories...")
    for idx, row in users_to_plot.iterrows():
        uid = row["uid"]

        for col in df.columns[1:]:  # Skip 'uid' column
            if col.startswith("t_"):
                timeline_idx = int(col[2:])  # Extract timeline index
                location = row[col]

                if pd.notna(location) and location != "":
                    try:
                        x, y = map(int, location.split(","))
                        # Only include valid coordinates (not 999,999 and within grid)
                        if 0 <= x <= 199 and 0 <= y <= 199:
                            trajectories[uid].append((timeline_idx, x, y))
                    except (ValueError, AttributeError):
                        continue

    print(f"Extracted trajectories for {len(trajectories)} users")

    # Sort trajectories by timeline for each user
    for uid in trajectories:
        trajectories[uid].sort(key=lambda x: x[0])  # Sort by timeline_idx

    # Generate distinct colors for users
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(trajectories))))
    if len(trajectories) > 20:
        # If more than 20 users, use more color maps
        colors = np.vstack(
            [
                plt.cm.tab20(np.linspace(0, 1, 20)),
                plt.cm.tab20b(np.linspace(0, 1, 20)),
                plt.cm.tab20c(np.linspace(0, 1, 20)),
                plt.cm.Set3(np.linspace(0, 1, 12)),
                plt.cm.Set1(np.linspace(0, 1, 9)),
            ]
        )

    # Create the main visualization
    plt.figure(figsize=(15, 15))

    plotted_users = 0
    legend_elements = []

    for i, uid in enumerate(user_ids):
        if uid not in trajectories or len(trajectories[uid]) < 2:
            continue

        traj = trajectories[uid]
        times = [t[0] for t in traj]
        xs = [t[1] for t in traj]
        ys = [t[2] for t in traj]

        color = colors[plotted_users % len(colors)]

        # Plot trajectory as connected line
        plt.plot(xs, ys, color=color, alpha=0.7, linewidth=1, marker="o", markersize=2)

        # Mark start point
        plt.plot(
            xs[0],
            ys[0],
            color=color,
            marker="s",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Mark end point
        plt.plot(
            xs[-1],
            ys[-1],
            color=color,
            marker="^",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Add to legend (only first 20 users to avoid clutter)
        if plotted_users < 20:
            legend_elements.append(
                plt.Line2D([0], [0], color=color, lw=2, label=f"User {uid}")
            )

        plotted_users += 1

    plt.xlim(0, 199)
    plt.ylim(0, 199)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.title(
        f"Mobility Paths of Top {plotted_users} Users\n(Squares=Start, Triangles=End)",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)

    # Add legend for first 20 users
    if legend_elements:
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
        )

    plt.tight_layout()

    if save_plots:
        plt.savefig("mobility_paths_overview.png", dpi=300, bbox_inches="tight")
        print("Saved: mobility_paths_overview.png")

    plt.show()

    # Create a density heatmap
    print("\nCreating density heatmap...")
    plt.figure(figsize=(12, 10))

    # Collect all points for density calculation
    all_points = []
    for uid in trajectories:
        for _, x, y in trajectories[uid]:
            all_points.append((x, y))

    if all_points:
        xs_all = [p[0] for p in all_points]
        ys_all = [p[1] for p in all_points]

        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            xs_all, ys_all, bins=50, range=[[0, 199], [0, 199]]
        )

        # Plot heatmap
        im = plt.imshow(
            heatmap.T,
            origin="lower",
            extent=[0, 199, 0, 199],
            cmap="YlOrRd",
            interpolation="bilinear",
        )
        plt.colorbar(im, label="Visit Frequency")
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.title(f"Mobility Density Heatmap - Top {plotted_users} Users", fontsize=14)

        if save_plots:
            plt.savefig("mobility_density_heatmap.png", dpi=300, bbox_inches="tight")
            print("Saved: mobility_density_heatmap.png")

        plt.show()

    # Create individual user plots (first 9 users)
    print("\nCreating individual user plots...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    individual_users = list(trajectories.keys())[:9]

    for i, uid in enumerate(individual_users):
        ax = axes[i]
        traj = trajectories[uid]

        if len(traj) >= 2:
            times = [t[0] for t in traj]
            xs = [t[1] for t in traj]
            ys = [t[2] for t in traj]

            # Plot trajectory with time-based coloring
            scatter = ax.scatter(xs, ys, c=times, cmap="viridis", s=20, alpha=0.7)
            ax.plot(xs, ys, color="gray", alpha=0.5, linewidth=1)

            # Mark start and end
            ax.plot(xs[0], ys[0], "go", markersize=8, label="Start")
            ax.plot(xs[-1], ys[-1], "ro", markersize=8, label="End")

            ax.set_xlim(0, 199)
            ax.set_ylim(0, 199)
            ax.set_title(f"User {uid}\n({len(traj)} points)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Add colorbar for time
            plt.colorbar(scatter, ax=ax, label="Timeline")
        else:
            ax.text(
                0.5,
                0.5,
                f"User {uid}\nInsufficient data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlim(0, 199)
            ax.set_ylim(0, 199)

    plt.tight_layout()

    if save_plots:
        plt.savefig("individual_user_paths.png", dpi=300, bbox_inches="tight")
        print("Saved: individual_user_paths.png")

    plt.show()

    # Print statistics
    print(f"\nStatistics:")
    print(f"Users plotted: {plotted_users}")
    print(f"Total trajectory points: {len(all_points)}")

    trajectory_lengths = [len(trajectories[uid]) for uid in trajectories]
    if trajectory_lengths:
        print(f"Avg points per user: {np.mean(trajectory_lengths):.1f}")
        print(f"Max points per user: {max(trajectory_lengths)}")
        print(f"Min points per user: {min(trajectory_lengths)}")

    # Show coordinate ranges used
    if all_points:
        xs_all = [p[0] for p in all_points]
        ys_all = [p[1] for p in all_points]
        print(f"X coordinate range: {min(xs_all)} to {max(xs_all)}")
        print(f"Y coordinate range: {min(ys_all)} to {max(ys_all)}")


if __name__ == "__main__":
    csv_file = "city_C_top_users_wide.csv"

    try:
        visualize_mobility_paths(csv_file, num_users=10, save_plots=True)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found!")
        print("Make sure you've run the converter script first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
