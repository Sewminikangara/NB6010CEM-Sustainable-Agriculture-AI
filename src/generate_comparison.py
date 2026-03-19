"""Generates a publication-quality bar chart comparing model evaluation metrics."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import config


def create_comparison_plot():
    models = ['MobileNetV2\n(Fine-tuned)', 'ViT-Tiny\n(Baseline)']
    metrics = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1-Score\n(Weighted)']

    # Updated evaluation results
    mobilenet_vals = [0.91, 0.92, 0.91, 0.91]
    vit_vals       = [0.34, 0.46, 0.34, 0.30]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f4f4f4')

    bars1 = ax.bar(x - width / 2, mobilenet_vals, width, label='MobileNetV2 (Fine-tuned)',
                   color='#27ae60', edgecolor='white', linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + width / 2, vit_vals, width, label='ViT-Tiny (Baseline)',
                   color='#2980b9', edgecolor='white', linewidth=0.8, zorder=3)

    # Value labels on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f'{h:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1a5e31')

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f'{h:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1a5278')

    ax.set_ylabel('Score', fontsize=12, labelpad=10)
    ax.set_title('Model Performance Comparison — 38-Class Plant Disease Classification',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(config.PLOT_SAVE_DIR, "comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    create_comparison_plot()
