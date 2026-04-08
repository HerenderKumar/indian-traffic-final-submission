import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Force Python to look in the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_comparison_charts():
    print("📊 Generating Performance Charts...")
    
    # Load the JSON data
    try:
        with open("baseline_scores.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ baseline_scores.json not found! Run inference.py first.")
        return

    metrics = data["metrics"]
    
    # Set up the visual style (Dark Mode)
    plt.style.use('dark_background')
    sns.set_palette(["#ef4444", "#34d399"]) # Red for baseline, Emerald for AI
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#0f172a') # Slate 900
    
    # --- Chart 1: Ambulance Wait Time ---
    labels = ['Standard Traffic Light', 'NexFlow AI']
    amb_values = [metrics["total_ambulance_wait_time"]["baseline"], metrics["total_ambulance_wait_time"]["ai"]]
    
    bars1 = ax1.bar(labels, amb_values, color=['#475569', '#34d399'], edgecolor='none')
    ax1.set_title("Ambulance Wait Time (Seconds)\nLower is Better", fontsize=14, pad=15, color='white')
    ax1.set_ylabel("Seconds", color='white')
    ax1.set_facecolor('#0f172a')
    ax1.grid(axis='y', alpha=0.2)
    
    # Add numbers on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval}s", ha='center', va='bottom', color='white', fontweight='bold')

    # --- Chart 2: Wasted Green Time ---
    waste_values = [metrics["wasted_green_time"]["baseline"], metrics["wasted_green_time"]["ai"]]
    
    bars2 = ax2.bar(labels, waste_values, color=['#475569', '#3b82f6'], edgecolor='none') # Slate and Blue
    ax2.set_title("Wasted Green Time (Seconds)\nLower is Better", fontsize=14, pad=15, color='white')
    ax2.set_facecolor('#0f172a')
    ax2.grid(axis='y', alpha=0.2)
    
    # Add numbers on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}s", ha='center', va='bottom', color='white', fontweight='bold')

    plt.tight_layout(pad=3.0)
    
    # Save the chart
    output_path = "performance_comparison.png"
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300)
    print(f"✅ Chart saved successfully as {output_path}")
    
    # Show the chart to the user
    plt.show()

if __name__ == "__main__":
    generate_comparison_charts()