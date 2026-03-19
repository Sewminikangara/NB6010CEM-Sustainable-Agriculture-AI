"""Plots training and validation accuracy/loss curves."""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import config


def create_training_curves():
    # Results recorded from the actual training run (10 epochs, subset=250)
    epochs = list(range(10))

    train_loss = [2.3537, 1.3925, 1.1791, 1.0853, 0.9796, 0.8463, 0.7722, 0.7595, 0.6890, 0.6529]
    val_loss   = [1.3248, 0.9229, 0.8655, 0.7527, 0.6597, 0.5692, 0.4983, 0.5315, 0.4795, 0.4205]

    train_acc  = [0.3508, 0.5903, 0.6637, 0.6859, 0.7000, 0.7319, 0.7358, 0.7399, 0.7647, 0.7699]
    val_acc    = [0.6171, 0.7523, 0.7658, 0.7643, 0.7793, 0.8093, 0.8213, 0.8078, 0.8138, 0.8393]

    fig = plt.figure(figsize=(13, 5))
    fig.patch.set_facecolor('#f9f9f9')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Loss Plot ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#f4f4f4')
    ax1.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Training Loss')
    ax1.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2, markersize=5, label='Validation Loss')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.set_xticks(epochs)
    ax1.legend(fontsize=10)
    ax1.grid(linestyle='--', alpha=0.5)
    ax1.spines[['top', 'right']].set_visible(False)

    # --- Accuracy Plot ---
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#f4f4f4')
    ax2.plot(epochs, [v * 100 for v in train_acc], 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Training Accuracy')
    ax2.plot(epochs, [v * 100 for v in val_acc], 's--', color='#27ae60', linewidth=2, markersize=5, label='Validation Accuracy')
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_xticks(epochs)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=10)
    ax2.grid(linestyle='--', alpha=0.5)
    ax2.spines[['top', 'right']].set_visible(False)

    fig.suptitle('MobileNetV2 Fine-Tuning — Training Dynamics (PlantVillage, 38 Classes)',
                 fontsize=13, fontweight='bold', y=1.02)

    save_path = os.path.join(config.PLOT_SAVE_DIR, "mobilenet_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    create_training_curves()
