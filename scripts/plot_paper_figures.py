import matplotlib.pyplot as plt
import numpy as np
import os

# Data matching Paper Table 1
methods = ['ERM', 'DANN', 'V-REx']
# Target F1 Scores
clean_scores = [0.85, 0.84, 0.81]
pois_scores = [0.83, 0.75, 0.82]

# Leakage (AUC)
leakage_auc = [0.78, 0.77, 0.79]

def plot_fig1_performance():
    # Grouped Bar Chart: Clean vs Poisoned
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, clean_scores, width, label='Clean Train', color='#2ca02c') # Green
    rects2 = ax.bar(x + width/2, pois_scores, width, label='Poisoned (SAST)', color='#d62728') # Red
    
    ax.set_ylabel('Target F1 Score')
    ax.set_title('Cross-Dataset Generalization Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig1_performance.png', dpi=300)
    print("Saved fig1_performance.png")

def plot_fig2_sast_drop():
    # Delta F1 (Drop)
    drops = np.array(pois_scores) - np.array(clean_scores)
    # drops: [-0.02, -0.09, +0.01]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, drops, color=['gray', 'firebrick', 'gray'])
    
    ax.set_ylabel('Performance Drop (Delta F1)')
    ax.set_title('SAST Vulnerability (Performance Drop under Poisoning)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-0.15, 0.05)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:+.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig2_sast_drop.png', dpi=300)
    print("Saved fig2_sast_drop.png")

def plot_fig3_leakage():
    # Leakage AUC
    # Baseline Chance = 0.5?
    # Actually leakage is binary (Source vs Target)? Yes.
    
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(methods, leakage_auc, color=['#1f77b4', '#ff7f0e', '#9467bd'])
    
    ax.set_ylabel('Domain Probe AUC')
    ax.set_title('Dataset-Identity Leakage')
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color='red', linestyle='--', label='Random Chance')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig3_leakage.png', dpi=300)
    print("Saved fig3_leakage.png")

if __name__ == "__main__":
    os.makedirs('paper/figures', exist_ok=True)
    plot_fig1_performance()
    plot_fig2_sast_drop()
    plot_fig3_leakage()
