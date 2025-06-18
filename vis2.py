import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import seaborn as sns


def parse_location(loc_str):
    """Parse location string like '97,152' into (x, y) tuple"""
    if pd.isna(loc_str) or loc_str == "":
        return None
    try:
        x, y = map(int, loc_str.split(","))
        return (x, y)
    except:
        return None


def extract_user_trajectory(user_row, training_slots, prediction_slots):
    """Extract trajectory data for a user"""
    trajectory = {
        "training": {"times": [], "locations": []},
        "prediction": {"times": [], "locations": []},
    }

    # Extract training period (days 1-60)
    for i, slot in enumerate(training_slots):
        loc_str = user_row[slot]
        loc = parse_location(loc_str)
        if loc:
            trajectory["training"]["times"].append(i)
            trajectory["training"]["locations"].append(loc)

    # Extract prediction period (days 61-75)
    for i, slot in enumerate(prediction_slots):
        loc_str = user_row[slot]
        loc = parse_location(loc_str)
        if loc:
            trajectory["prediction"]["times"].append(len(training_slots) + i)
            trajectory["prediction"]["locations"].append(loc)

    return trajectory


def create_user_plot(user_row, user_idx, training_slots, prediction_slots, output_dir):
    """Create visualization for a single user"""
    uid = user_row["uid"]
    trajectory = extract_user_trajectory(user_row, training_slots, prediction_slots)

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f"User {uid} - Mobility Pattern (User #{user_idx+1}/100)",
        fontsize=16,
        fontweight="bold",
    )

    # Determine if user has prediction data
    has_prediction = len(trajectory["prediction"]["locations"]) > 0
    user_type = "WITH prediction data" if has_prediction else "WITHOUT prediction data"

    # --- LEFT PLOT: Spatial Path ---
    ax1.set_title(f"Spatial Movement Path\n({user_type})", fontsize=12, pad=20)

    if trajectory["training"]["locations"]:
        # Training period path
        train_x = [loc[0] for loc in trajectory["training"]["locations"]]
        train_y = [loc[1] for loc in trajectory["training"]["locations"]]

        # Plot training path
        ax1.plot(
            train_x, train_y, "b-", alpha=0.6, linewidth=1, label="Training (Days 1-60)"
        )
        ax1.scatter(train_x, train_y, c="blue", s=2, alpha=0.7)

        # Mark start and end of training
        ax1.scatter(
            train_x[0],
            train_y[0],
            c="green",
            s=100,
            marker="s",
            label="Training Start",
            edgecolor="black",
            linewidth=1,
        )
        ax1.scatter(
            train_x[-1],
            train_y[-1],
            c="orange",
            s=100,
            marker="D",
            label="Training End",
            edgecolor="black",
            linewidth=1,
        )

    if trajectory["prediction"]["locations"]:
        # Prediction period path
        pred_x = [loc[0] for loc in trajectory["prediction"]["locations"]]
        pred_y = [loc[1] for loc in trajectory["prediction"]["locations"]]

        # Plot prediction path
        ax1.plot(
            pred_x,
            pred_y,
            "r-",
            alpha=0.8,
            linewidth=2,
            label="Prediction (Days 61-75)",
        )
        ax1.scatter(pred_x, pred_y, c="red", s=8, alpha=0.8)

        # Mark prediction start
        ax1.scatter(
            pred_x[0],
            pred_y[0],
            c="purple",
            s=100,
            marker="^",
            label="Prediction Start",
            edgecolor="black",
            linewidth=1,
        )

    ax1.set_xlabel("X Coordinate", fontsize=11)
    ax1.set_ylabel("Y Coordinate", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 199)
    ax1.set_ylim(0, 199)
    ax1.set_aspect("equal")

    # --- RIGHT PLOT: Timeline ---
    ax2.set_title("Movement Timeline\n(Time vs Location)", fontsize=12, pad=20)

    if trajectory["training"]["locations"]:
        train_times = trajectory["training"]["times"]
        train_x = [loc[0] for loc in trajectory["training"]["locations"]]
        train_y = [loc[1] for loc in trajectory["training"]["locations"]]

        # Plot X and Y coordinates over time
        ax2.plot(
            train_times,
            train_x,
            "b-",
            alpha=0.7,
            linewidth=1,
            label="X coordinate (Training)",
        )
        ax2.plot(
            train_times,
            train_y,
            "c-",
            alpha=0.7,
            linewidth=1,
            label="Y coordinate (Training)",
        )

    if trajectory["prediction"]["locations"]:
        pred_times = trajectory["prediction"]["times"]
        pred_x = [loc[0] for loc in trajectory["prediction"]["locations"]]
        pred_y = [loc[1] for loc in trajectory["prediction"]["locations"]]

        # Plot prediction coordinates
        ax2.plot(
            pred_times,
            pred_x,
            "r-",
            alpha=0.8,
            linewidth=2,
            label="X coordinate (Prediction)",
        )
        ax2.plot(
            pred_times,
            pred_y,
            "m-",
            alpha=0.8,
            linewidth=2,
            label="Y coordinate (Prediction)",
        )

    # Mark the transition point
    training_end = len(training_slots) - 1
    ax2.axvline(
        x=training_end,
        color="orange",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="Training/Prediction Split",
    )

    ax2.set_xlabel("Time Slot", fontsize=11)
    ax2.set_ylabel("Coordinate Value", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 199)

    # Add statistics text box
    stats_text = f"""
    Training points: {len(trajectory['training']['locations'])}
    Prediction points: {len(trajectory['prediction']['locations'])}
    Total timespan: {len(training_slots) + len(prediction_slots)} slots
    User type: {user_type}
    """

    ax2.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save the plot
    output_file = output_dir / f"user_{uid:03d}_path.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def create_summary_plot(df, training_slots, prediction_slots, output_dir):
    """Create a summary visualization showing all users"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "Summary: All 100 Users Mobility Patterns", fontsize=16, fontweight="bold"
    )

    users_no_pred = []
    users_with_pred = []

    # Categorize users and collect data
    for idx, row in df.iterrows():
        uid = row["uid"]
        trajectory = extract_user_trajectory(row, training_slots, prediction_slots)

        if len(trajectory["prediction"]["locations"]) > 0:
            users_with_pred.append((uid, trajectory))
        else:
            users_no_pred.append((uid, trajectory))

    # Plot 1: All users spatial paths (first 10 vs rest)
    ax1.set_title("Spatial Paths: First 10 vs Rest 90 Users", fontsize=12)

    # Plot first 10 users (no prediction data)
    for uid, traj in users_no_pred[:10]:
        if traj["training"]["locations"]:
            x_coords = [loc[0] for loc in traj["training"]["locations"]]
            y_coords = [loc[1] for loc in traj["training"]["locations"]]
            ax1.plot(x_coords, y_coords, "b-", alpha=0.3, linewidth=0.5)

    # Plot sample of remaining users (with prediction data)
    for uid, traj in users_with_pred[:20]:  # Show first 20 for clarity
        if traj["training"]["locations"]:
            x_coords = [loc[0] for loc in traj["training"]["locations"]]
            y_coords = [loc[1] for loc in traj["training"]["locations"]]
            ax1.plot(x_coords, y_coords, "r-", alpha=0.3, linewidth=0.5)

    ax1.plot([], [], "b-", label="Users 1-10 (No prediction data)", linewidth=2)
    ax1.plot([], [], "r-", label="Users 11-100 (With prediction data)", linewidth=2)
    ax1.set_xlim(0, 199)
    ax1.set_ylim(0, 199)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Data availability heatmap
    ax2.set_title("Data Availability Heatmap (100 Users × Time)", fontsize=12)

    availability_matrix = []
    for idx, row in df.iterrows():
        user_availability = []
        for slot in training_slots + prediction_slots:
            has_data = 1 if pd.notna(row[slot]) and row[slot] != "" else 0
            user_availability.append(has_data)
        availability_matrix.append(user_availability)

    im = ax2.imshow(
        availability_matrix, aspect="auto", cmap="RdYlBu_r", interpolation="nearest"
    )
    ax2.set_xlabel("Time Slot")
    ax2.set_ylabel("User Index")
    ax2.axvline(x=len(training_slots) - 1, color="white", linestyle="--", linewidth=2)
    plt.colorbar(im, ax=ax2, label="Data Available")

    # Plot 3: Location distribution
    ax3.set_title("Location Distribution (All Training Data)", fontsize=12)

    all_locations = []
    for idx, row in df.iterrows():
        for slot in training_slots:
            loc_str = row[slot]
            loc = parse_location(loc_str)
            if loc:
                all_locations.append(loc)

    if all_locations:
        x_coords = [loc[0] for loc in all_locations]
        y_coords = [loc[1] for loc in all_locations]

        ax3.hist2d(x_coords, y_coords, bins=50, cmap="YlOrRd")
        ax3.set_xlabel("X Coordinate")
        ax3.set_ylabel("Y Coordinate")
        ax3.set_xlim(0, 199)
        ax3.set_ylim(0, 199)

    # Plot 4: Statistics
    ax4.set_title("Dataset Statistics", fontsize=12)
    ax4.axis("off")

    stats_text = f"""
    DATASET SUMMARY
    
    Total Users: {len(df)}
    Users without prediction data: {len(users_no_pred)} (First 10)
    Users with prediction data: {len(users_with_pred)} (Remaining 90)
    
    TIME PERIODS:
    Training period: Days 1-60 ({len(training_slots)} time slots)
    Prediction period: Days 61-75 ({len(prediction_slots)} time slots)
    
    LOCATION COVERAGE:
    Total unique locations: {len(set(all_locations)) if all_locations else 0}
    Grid size: 200 × 200 (coordinates 0-199)
    
    DATA QUALITY:
    Total location observations: {len(all_locations):,}
    Average per user (training): {len(all_locations)/len(df):.1f}
    """

    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    # Save summary
    summary_file = output_dir / "summary_all_users.png"
    plt.savefig(summary_file, dpi=300, bbox_inches="tight")
    plt.close()

    return summary_file


def visualize_all_users(data_file, output_dir="user_visualizations"):
    """Create visualizations for all users"""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load data
    print("Loading processed data...")
    df = pd.read_csv(data_file)

    # Define time periods
    training_slots = [f"t_{i}" for i in range(0, 60 * 48)]
    prediction_slots = [f"t_{i}" for i in range(60 * 48, 75 * 48)]

    print(f"Creating visualizations for {len(df)} users...")
    print(f"Output directory: {output_path.absolute()}")

    # Create individual user plots
    created_files = []
    for idx, (_, user_row) in enumerate(df.iterrows()):
        output_file = create_user_plot(
            user_row, idx, training_slots, prediction_slots, output_path
        )
        created_files.append(output_file)

        if (idx + 1) % 10 == 0:
            print(f"Created {idx + 1}/{len(df)} user visualizations...")

    # Create summary plot
    print("Creating summary visualization...")
    summary_file = create_summary_plot(
        df, training_slots, prediction_slots, output_path
    )
    created_files.append(summary_file)

    print(f"\n✅ All visualizations completed!")
    print(f"📁 Created {len(created_files)} files in: {output_path.absolute()}")
    print(f"📊 Summary plot: {summary_file.name}")

    return created_files


def main():
    # Configuration
    input_file = "city_C_processed.csv"  # Your processed data file

    if not Path(input_file).exists():
        print(f"❌ Error: Input file '{input_file}' not found!")
        print("Please run the data processing script first.")
        return

    # Create visualizations
    visualize_all_users(input_file)


if __name__ == "__main__":
    main()
