import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from collections import defaultdict


def forward_fill_locations(user_data):
    """
    Forward fill missing locations - if no data, user stays at last known position
    """
    # Sort by timeline
    user_data = user_data.sort_values('timeline_idx').copy()
    
    # Forward fill locations
    user_data['x_filled'] = user_data['x'].replace(999, np.nan)
    user_data['y_filled'] = user_data['y'].replace(999, np.nan)
    
    # Forward fill
    user_data['x_filled'] = user_data['x_filled'].fillna(method='ffill')
    user_data['y_filled'] = user_data['y_filled'].fillna(method='ffill')
    
    # If still NaN (no previous position), use first valid position
    first_valid_x = user_data['x_filled'].dropna().iloc[0] if len(user_data['x_filled'].dropna()) > 0 else 0
    first_valid_y = user_data['y_filled'].dropna().iloc[0] if len(user_data['y_filled'].dropna()) > 0 else 0
    
    user_data['x_filled'] = user_data['x_filled'].fillna(first_valid_x)
    user_data['y_filled'] = user_data['y_filled'].fillna(first_valid_y)
    
    return user_data


def convert_city_with_forward_fill(input_file, output_file, max_users=100, debug=True):
    """
    Convert city data with forward fill approach - missing data means staying at same position
    """
    
    print("Loading city data...")
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique users: {df['uid'].nunique()}")
    print(f"Date range: {df['d'].min()} to {df['d'].max()}")
    print(f"Time slot range: {df['t'].min()} to {df['t'].max()}")
    
    # Calculate timeline index
    df['timeline_idx'] = (df['d'] - 1) * 48 + df['t']
    
    # Count records per user (including missing positions)
    user_counts = df['uid'].value_counts()
    print(f"\nUser record statistics:")
    print(f"Max records: {user_counts.max()}")
    print(f"Min records: {user_counts.min()}")
    print(f"Avg records: {user_counts.mean():.1f}")
    
    # Select top users by total record count
    top_users = user_counts.head(max_users).index.tolist()
    print(f"\nSelected top {len(top_users)} users by record count")
    
    # Filter to selected users
    df_filtered = df[df['uid'].isin(top_users)].copy()
    
    print(f"Processing {len(top_users)} users with forward fill...")
    
    # Process each user with forward fill
    processed_users = []
    
    for uid in top_users:
        user_data = df_filtered[df_filtered['uid'] == uid].copy()
        
        # Create complete timeline for this user (all possible timestamps)
        complete_timeline = pd.DataFrame({
            'timeline_idx': range(75 * 48),  # 0 to 3599
            'uid': uid,
            'd': [(i // 48) + 1 for i in range(75 * 48)],
            't': [i % 48 for i in range(75 * 48)]
        })
        
        # Merge with actual data
        user_complete = complete_timeline.merge(
            user_data[['timeline_idx', 'x', 'y']], 
            on='timeline_idx', 
            how='left'
        )
        
        # Apply forward fill
        user_filled = forward_fill_locations(user_complete)
        
        # Create location string
        user_filled['location'] = user_filled.apply(
            lambda row: f"{int(row['x_filled'])},{int(row['y_filled'])}", axis=1
        )
        
        processed_users.append(user_filled[['uid', 'timeline_idx', 'location']])
    
    # Combine all users
    all_data = pd.concat(processed_users, ignore_index=True)
    
    print(f"Created complete data for {len(top_users)} users")
    
    # Convert to wide format
    print("Converting to wide format...")
    wide_data = all_data.pivot(index='uid', columns='timeline_idx', values='location')
    
    # Rename columns to t_0, t_1, etc.
    wide_data.columns = [f't_{i}' for i in range(75 * 48)]
    
    # Reset index
    wide_data.reset_index(inplace=True)
    
    print(f"Wide format shape: {wide_data.shape}")
    
    # Calculate statistics
    total_cells = len(wide_data) * (75 * 48)
    filled_cells = wide_data.iloc[:, 1:].notna().sum().sum()
    fill_rate = filled_cells / total_cells * 100
    
    print(f"\nStatistics:")
    print(f"Users: {len(wide_data)}")
    print(f"Timeline slots: {75 * 48}")
    print(f"Fill rate: {fill_rate:.2f}% (should be 100% with forward fill)")
    
    # Show sample
    print(f"\nSample data (first 3 users, timeline slots 0-10):")
    sample_cols = ['uid'] + [f't_{i}' for i in range(11)]
    print(wide_data[sample_cols].head(3))
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    wide_data.to_csv(output_file, index=False)
    
    print("Conversion completed!")
    return wide_data


def analyze_user_movement_patterns(wide_df, user_id, save_plots=True):
    """
    Analyze movement patterns for a specific user
    """
    user_data = wide_df[wide_df['uid'] == user_id].iloc[0]
    
    # Extract locations
    locations = []
    for i in range(75 * 48):
        loc_str = user_data[f't_{i}']
        x, y = map(int, loc_str.split(','))
        day = (i // 48) + 1
        timeslot = i % 48
        locations.append((i, day, timeslot, x, y))
    
    locations_df = pd.DataFrame(locations, columns=['timeline_idx', 'day', 'timeslot', 'x', 'y'])
    
    print(f"\nAnalyzing user {user_id} movement patterns:")
    print(f"Unique locations visited: {len(locations_df[['x', 'y']].drop_duplicates())}")
    print(f"Total distance traveled: {calculate_total_distance(locations_df):.1f}")
    
    # Visualize daily patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Trajectory plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(locations_df['x'], locations_df['y'], 
                         c=locations_df['timeline_idx'], cmap='viridis', 
                         s=10, alpha=0.7)
    ax1.plot(locations_df['x'], locations_df['y'], 'k-', alpha=0.3, linewidth=0.5)
    ax1.set_title(f'User {user_id} - Full Trajectory')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    plt.colorbar(scatter, ax=ax1, label='Timeline')
    
    # 2. Daily activity heatmap
    ax2 = axes[0, 1]
    activity_matrix = locations_df.pivot_table(
        values='x', index='day', columns='timeslot', aggfunc='count', fill_value=0
    )
    sns.heatmap(activity_matrix, ax=ax2, cmap='YlOrRd', cbar_kws={'label': 'Activity'})
    ax2.set_title(f'User {user_id} - Daily Activity Pattern')
    ax2.set_xlabel('Time Slot (30min intervals)')
    ax2.set_ylabel('Day')
    
    # 3. Distance from home over time
    ax3 = axes[1, 0]
    home_x, home_y = locations_df.iloc[0]['x'], locations_df.iloc[0]['y']  # First location as "home"
    locations_df['distance_from_home'] = np.sqrt(
        (locations_df['x'] - home_x)**2 + (locations_df['y'] - home_y)**2
    )
    ax3.plot(locations_df['timeline_idx'], locations_df['distance_from_home'])
    ax3.set_title(f'User {user_id} - Distance from Home Over Time')
    ax3.set_xlabel('Timeline Index')
    ax3.set_ylabel('Distance from Home')
    
    # 4. Hourly movement patterns
    ax4 = axes[1, 1]
    locations_df['hour'] = (locations_df['timeslot'] * 0.5).astype(int)  # Convert to hour
    hourly_movement = locations_df.groupby('hour')['distance_from_home'].mean()
    ax4.bar(hourly_movement.index, hourly_movement.values)
    ax4.set_title(f'User {user_id} - Average Distance by Hour')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Avg Distance from Home')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'user_{user_id}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: user_{user_id}_analysis.png")
    
    plt.show()
    
    return locations_df


def calculate_total_distance(locations_df):
    """
    Calculate total distance traveled by user
    """
    total_distance = 0
    for i in range(1, len(locations_df)):
        x1, y1 = locations_df.iloc[i-1][['x', 'y']]
        x2, y2 = locations_df.iloc[i][['x', 'y']]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    return total_distance


def create_day_similarity_matrix(wide_df, user_id, metric='geobleu_simple'):
    """
    Create similarity matrix between days for a user
    As described in ideas.md: Tag = 48 timestamps
    """
    user_data = wide_df[wide_df['uid'] == user_id].iloc[0]
    
    # Extract daily trajectories (48 timestamps per day)
    daily_trajectories = []
    for day in range(1, 76):  # Days 1-75
        day_start = (day - 1) * 48
        day_end = day * 48
        day_locations = []
        
        for i in range(day_start, day_end):
            loc_str = user_data[f't_{i}']
            x, y = map(int, loc_str.split(','))
            day_locations.append((x, y))
        
        daily_trajectories.append(day_locations)
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((75, 75))
    
    for i in range(75):
        for j in range(75):
            if metric == 'geobleu_simple':
                # Simple GEOBLEU-like metric: count of matching locations
                traj1, traj2 = daily_trajectories[i], daily_trajectories[j]
                matches = sum(1 for a, b in zip(traj1, traj2) if a == b)
                similarity = matches / 48.0  # Normalize by day length
            else:
                # Euclidean distance based similarity
                traj1, traj2 = daily_trajectories[i], daily_trajectories[j]
                distances = [np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) for a, b in zip(traj1, traj2)]
                avg_distance = np.mean(distances)
                similarity = 1.0 / (1.0 + avg_distance)  # Inverse distance similarity
            
            similarity_matrix[i, j] = similarity
    
    return similarity_matrix, daily_trajectories


def visualize_day_similarity_heatmap(wide_df, user_ids, save_plots=True):
    """
    Create day similarity heatmaps as described in ideas.md
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, user_id in enumerate(user_ids[:4]):
        ax = axes[idx]
        
        similarity_matrix, _ = create_day_similarity_matrix(wide_df, user_id)
        
        sns.heatmap(similarity_matrix, ax=ax, cmap='viridis', 
                   cbar_kws={'label': 'Similarity'})
        ax.set_title(f'User {user_id} - Day Similarity Matrix')
        ax.set_xlabel('Day')
        ax.set_ylabel('Day')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('day_similarity_heatmaps.png', dpi=300, bbox_inches='tight')
        print("Saved: day_similarity_heatmaps.png")
    
    plt.show()


def apply_dbscan_clustering(wide_df, user_ids, eps=5, min_samples=10):
    """
    Apply DBSCAN clustering to user trajectories
    """
    clustering_results = {}
    
    for user_id in user_ids:
        user_data = wide_df[wide_df['uid'] == user_id].iloc[0]
        
        # Extract all locations
        locations = []
        for i in range(75 * 48):
            loc_str = user_data[f't_{i}']
            x, y = map(int, loc_str.split(','))
            locations.append([x, y])
        
        locations_array = np.array(locations)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(locations_array)
        
        clustering_results[user_id] = {
            'locations': locations_array,
            'clusters': clusters,
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
            'n_noise': list(clusters).count(-1)
        }
        
        print(f"User {user_id}: {clustering_results[user_id]['n_clusters']} clusters, "
              f"{clustering_results[user_id]['n_noise']} noise points")
    
    return clustering_results


def visualize_clusters(clustering_results, save_plots=True):
    """
    Visualize DBSCAN clustering results
    """
    n_users = len(clustering_results)
    cols = min(4, n_users)
    rows = (n_users + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_users == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (user_id, results) in enumerate(clustering_results.items()):
        ax = axes[idx] if n_users > 1 else axes[0]
        
        locations = results['locations']
        clusters = results['clusters']
        
        # Plot clusters
        unique_clusters = set(clusters)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster_id, color in zip(unique_clusters, colors):
            if cluster_id == -1:
                # Noise points in black
                mask = clusters == cluster_id
                ax.scatter(locations[mask, 0], locations[mask, 1], 
                          c='black', s=1, alpha=0.5, label='Noise')
            else:
                mask = clusters == cluster_id
                ax.scatter(locations[mask, 0], locations[mask, 1], 
                          c=[color], s=10, alpha=0.7, label=f'Cluster {cluster_id}')
        
        ax.set_title(f'User {user_id} - DBSCAN Clusters\n'
                    f'{results["n_clusters"]} clusters, {results["n_noise"]} noise')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(clustering_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('dbscan_clusters.png', dpi=300, bbox_inches='tight')
        print("Saved: dbscan_clusters.png")
    
    plt.show()


if __name__ == "__main__":
    input_file = "city_C_challengedata.csv"
    output_file = "city_C_forward_filled.csv"
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
    else:
        # Step 0: Convert data with forward fill (empty = same position)
        print("=== Step 0: Data completion (empty = same position) ===")
        wide_df = convert_city_with_forward_fill(input_file, output_file, max_users=20)
        
        # Get first few users for analysis
        user_ids = wide_df['uid'].head(4).tolist()
        print(f"\nAnalyzing users: {user_ids}")
        
        # Step 1: Graphics per user with path
        print("\n=== Step 1: Graphics per user with path ===")
        for user_id in user_ids[:2]:  # Analyze first 2 users in detail
            analyze_user_movement_patterns(wide_df, user_id)
        
        # Step 2: Day similarity heatmap (GEOBLEU between days)
        print("\n=== Step 2: Day similarity heatmap ===")
        visualize_day_similarity_heatmap(wide_df, user_ids)
        
        # Step 3: DBSCAN clustering
        print("\n=== Step 3: DBSCAN clustering ===")
        clustering_results = apply_dbscan_clustering(wide_df, user_ids[:4])
        visualize_clusters(clustering_results)
        
        print("\n=== Analysis completed! ===")
        print("Generated files:")
        print("- city_C_forward_filled.csv (wide format data)")
        print("- user_*_analysis.png (individual user analysis)")
        print("- day_similarity_heatmaps.png (day similarity matrices)")
        print("- dbscan_clusters.png (DBSCAN clustering results)")
