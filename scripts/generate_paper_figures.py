
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set style
sns.set(style="whitegrid", context="paper", font_scale=1.2)
OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_main_results():
    """Figure 1: Main Performance (Bar Plot)"""
    # Data from Table I
    data = {
        'Method': ['ERM', 'ERM', 'DANN', 'DANN', 'V-REx', 'V-REx', 'IRM', 'IRM'],
        'Condition': ['Clean', 'Poisoned', 'Clean', 'Poisoned', 'Clean', 'Poisoned', 'Clean', 'Poisoned'],
        'Target F1': [0.85, 0.83, 0.84, 0.75, 0.81, 0.82, 0.19, 0.16]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Method", y="Target F1", hue="Condition", data=df, palette=["#2ecc71", "#e74c3c"])
    plt.title("Figure 1: Cross-Dataset Generalization Performance")
    plt.ylim(0, 1.0)
    plt.ylabel("Target F1 Score (Chapman)")
    
    # Add annotations
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_performance.png", dpi=300)
    print("Generated Figure 1")

def plot_sast_drop():
    """Figure 2: SAST Vulnerability (Delta F1)"""
    methods = ['ERM', 'DANN', 'V-REx', 'IRM']
    drops = [-0.02, -0.09, +0.01, -0.03]
    colors = ['#95a5a6' if abs(x) < 0.05 else '#e74c3c' for x in drops]
    
    plt.figure(figsize=(8, 5))
    plt.bar(methods, drops, color=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Figure 2: SAST Vulnerability (Performance Drop under Shortcut)")
    plt.ylabel("Δ F1 (Poisoned - Clean)")
    plt.ylim(-0.15, 0.05)
    
    # Add labels
    for i, v in enumerate(drops):
        plt.text(i, v - 0.01 if v < 0 else v + 0.005, f"{v:+.2f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_sast_drop.png", dpi=300)
    print("Generated Figure 2")

def plot_leakage():
    """Figure 3: Dataset-Identity Leakage"""
    methods = ['ERM', 'DANN', 'V-REx', 'IRM']
    leakage = [99.8, 99.9, 99.7, 94.7]
    chance_level = 50.0
    
    plt.figure(figsize=(8, 5))
    plt.bar(methods, leakage, color="#34495e")
    plt.axhline(chance_level, color='red', linestyle='--', label='Random Chance (Invariant Ideal)')
    plt.title("Figure 3: Dataset-Identity Leakage (Linear Probe Accuracy)")
    plt.ylabel("Probe Accuracy (%)")
    plt.ylim(40, 105)
    plt.legend()
    
    for i, v in enumerate(leakage):
        plt.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_leakage.png", dpi=300)
    print("Generated Figure 3") # Fixed comment

if __name__ == "__main__":
    plot_main_results()
    plot_sast_drop()
    plot_leakage()
