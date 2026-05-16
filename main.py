import sys
sys.platform = "win32"
import platform
platform.machine = lambda: "AMD64"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_and_clean_data():
    """Handles folder logic, reads from data folder, and exports the cleaned CSV."""
    data_folder = "data"
    original_file = os.path.join(data_folder, "datasets_original.csv")
    cleaned_file = os.path.join(data_folder, "datasets_cleaned.csv")
    
    if not os.path.exists(original_file):
        raise FileNotFoundError(
            f"Could not find '{original_file}'.\n"
            f"Please ensure your original CSV is inside the 'data' folder!"
        )
    
    # Read the raw data
    df = pd.read_csv(original_file)
    
    # Drop rows missing vital plotting metrics
    required_metrics = ['Braking_Performance', 'Fuel_Efficiency', 'Engine_Performance', 'Emissions']
    df_cleaned = df.dropna(subset=required_metrics).copy()
    
    # Export cleaned CSV back to the data folder
    df_cleaned.to_csv(cleaned_file, index=False)
    print(f"[Saved] Cleaned dataset exported successfully to: '{cleaned_file}'")
    return df_cleaned

def generate_first_gif(df, output_folder):
    """GIF 1: Braking Performance vs Fuel Efficiency"""
    x_metric = 'Braking_Performance'
    y_metric = 'Fuel_Efficiency'
    
    df_sorted = df.sort_values(by=x_metric).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#f7f9fc')
    ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
    
    ax.set_xlim(df_sorted[x_metric].min() * 0.9, df_sorted[x_metric].max() * 1.1)
    ax.set_ylim(df_sorted[y_metric].min() * 0.9, df_sorted[y_metric].max() * 1.1)
    
    ax.set_title("Automotive Telemetry Limits Reconstruction Loop", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Braking Performance (Stopping Power)", fontsize=10)
    ax.set_ylabel("Fuel Efficiency (MPG)", fontsize=10)
    
    scatter = ax.scatter([], [], color='#1a73e8', alpha=0.6, edgecolor='white', s=50, label='Inspected Fleet Data')
    line, = ax.plot([], [], color='#e65100', linewidth=3, label='Fitted Fleet Limit Trend')
    ax.legend(loc="upper right")
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        line.set_data([], [])
        return scatter, line

    def update(frame):
        sub_df = df_sorted.iloc[:frame]
        x_vals = sub_df[x_metric].values
        y_vals = sub_df[y_metric].values
        scatter.set_offsets(np.c_[x_vals, y_vals])
        if len(sub_df) > 5:
            fit = np.poly1d(np.polyfit(x_vals, y_vals, 2))
            line.set_data(x_vals, fit(x_vals))
        return scatter, line

    total_frames = min(80, len(df_sorted))
    frame_indices = np.linspace(1, len(df_sorted) - 1, total_frames, dtype=int)
    
    print("Compiling GIF 1: Vehicle Dynamics Loop...")
    ani = animation.FuncAnimation(fig, update, frames=frame_indices, init_func=init, blit=True, interval=70)
    
    output_filepath = os.path.join(output_folder, 'vehicle_dynamics_animation.gif')
    ani.save(output_filepath, writer='pillow', fps=14)
    plt.close()
    print(f"[Saved] GIF 1 exported to '{output_filepath}'")

def generate_second_gif(df, output_folder):
    """GIF 2: Engine Performance vs Emissions Profile"""
    x_metric = 'Engine_Performance'
    y_metric = 'Emissions'
    
    # Sort by engine performance for sequential plotting
    df_sorted = df.sort_values(by=x_metric).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#fcfbf7')  # Light warm background
    ax.grid(True, linestyle='--', alpha=0.6, color='#dddddd')
    
    ax.set_xlim(df_sorted[x_metric].min() * 0.9, df_sorted[x_metric].max() * 1.1)
    ax.set_ylim(df_sorted[y_metric].min() * 0.9, df_sorted[y_metric].max() * 1.1)
    
    ax.set_title("Powertrain Output vs Emissions Signature Progression", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Engine Performance Index", fontsize=10)
    ax.set_ylabel("Carbon Emissions Rating", fontsize=10)
    
    # High-contrast Teal scatter dots with Crimson trend line
    scatter = ax.scatter([], [], color='#008080', alpha=0.6, edgecolor='white', s=50, label='Engine Nodes')
    line, = ax.plot([], [], color='#b30000', linewidth=3, label='Emissions Trendline')
    ax.legend(loc="upper left")
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        line.set_data([], [])
        return scatter, line

    def update(frame):
        sub_df = df_sorted.iloc[:frame]
        x_vals = sub_df[x_metric].values
        y_vals = sub_df[y_metric].values
        scatter.set_offsets(np.c_[x_vals, y_vals])
        if len(sub_df) > 5:
            fit = np.poly1d(np.polyfit(x_vals, y_vals, 2))
            line.set_data(x_vals, fit(x_vals))
        return scatter, line

    total_frames = min(80, len(df_sorted))
    frame_indices = np.linspace(1, len(df_sorted) - 1, total_frames, dtype=int)
    
    print("Compiling GIF 2: Engine Emissions Loop...")
    ani = animation.FuncAnimation(fig, update, frames=frame_indices, init_func=init, blit=True, interval=70)
    
    output_filepath = os.path.join(output_folder, 'engine_emissions_animation.gif')
    ani.save(output_filepath, writer='pillow', fps=14)
    plt.close()
    print(f"[Saved] GIF 2 exported to '{output_filepath}'")

if __name__ == "__main__":
    try:
        # 1. Clean data and sync to data folder
        cleaned_vehicle_data = load_and_clean_data()
        
        # 2. Setup outputs workspace
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. Compile both animations
        generate_first_gif(cleaned_vehicle_data, output_dir)
        generate_second_gif(cleaned_vehicle_data, output_dir)
        
        print("\n=== SUCCESS: Both animated tracking loops saved cleanly inside 'outputs/'! ===")
        
    except Exception as e:
        print(f"\n[Execution Error]: {e}")