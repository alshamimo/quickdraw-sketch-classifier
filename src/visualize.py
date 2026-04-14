import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch


# ─── Ensure Directory ────────────────────────────────────────────────────────
# Saves to the "plots" folder in the parent directory
os.makedirs("plots", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Loss and Accuracy Curves
# Shows the development of Loss and Accuracy per Epoch.
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(nn_history, cnn_history, save_path="plots/training_curves.png"):
    """
    Plots Loss and Accuracy curves for NN and CNN side by side.

    Args:
        nn_history:  dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        cnn_history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path:   Path to save the plot
    """

    # 2 rows, 2 columns = 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # x-axis: Epoch numbers (1 to N)
    nn_epochs  = range(1, len(nn_history['train_loss'])  + 1)
    cnn_epochs = range(1, len(cnn_history['train_loss']) + 1)

    # ── Plot [0,0]: NN Loss ───────────────────────────────────────────────────
    axes[0, 0].plot(nn_epochs, nn_history['train_loss'],
                    label='Train Loss', color='#3498db', linewidth=2)
    axes[0, 0].plot(nn_epochs, nn_history['val_loss'],
                    label='Val Loss',   color='#e74c3c', linewidth=2, linestyle='--')
    axes[0, 0].set_title('NN -- Loss', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ── Plot [0,1]: CNN Loss ──────────────────────────────────────────────────
    axes[0, 1].plot(cnn_epochs, cnn_history['train_loss'],
                    label='Train Loss', color='#2ecc71', linewidth=2)
    axes[0, 1].plot(cnn_epochs, cnn_history['val_loss'],
                    label='Val Loss',   color='#e67e22', linewidth=2, linestyle='--')
    axes[0, 1].set_title('CNN -- Loss', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ── Plot [1,0]: NN Accuracy ───────────────────────────────────────────────
    axes[1, 0].plot(nn_epochs, nn_history['train_acc'],
                    label='Train Acc', color='#3498db', linewidth=2)
    axes[1, 0].plot(nn_epochs, nn_history['val_acc'],
                    label='Val Acc',   color='#e74c3c', linewidth=2, linestyle='--')
    axes[1, 0].set_title('NN -- Accuracy', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ── Plot [1,1]: CNN Accuracy ──────────────────────────────────────────────
    axes[1, 1].plot(cnn_epochs, cnn_history['train_acc'],
                    label='Train Acc', color='#2ecc71', linewidth=2)
    axes[1, 1].plot(cnn_epochs, cnn_history['val_acc'],
                    label='Val Acc',   color='#e67e22', linewidth=2, linestyle='--')
    axes[1, 1].set_title('CNN -- Accuracy', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training Progress: NN vs. CNN', fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Accuracy Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_comparison(nn_acc, cnn_acc, save_path="plots/accuracy_comparison.png"):
    """
    Bar chart comparing final test accuracy.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_nn  = ax.bar(0,   nn_acc  * 100, width=0.35,
                     color='#3498db', label='NN',
                     alpha=0.85, edgecolor='white', linewidth=1.5)
    bar_cnn = ax.bar(0.5, cnn_acc * 100, width=0.35,
                     color='#2ecc71', label='CNN',
                     alpha=0.85, edgecolor='white', linewidth=1.5)

    ax.bar_label(bar_nn,  fmt='%.1f%%', padding=5, fontsize=13, fontweight='bold')
    ax.bar_label(bar_cnn, fmt='%.1f%%', padding=5, fontsize=13, fontweight='bold')

    diff = (cnn_acc - nn_acc) * 100
    ax.text(0.25, max(nn_acc, cnn_acc) * 100 + 8,
            f'CNN better by {diff:.1f}%',
            ha='center', fontsize=11, color='#2c3e50', fontstyle='italic')

    ax.set_title('Test Accuracy: NN vs. CNN', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_xticks([0, 0.5])
    ax.set_xticklabels(['NN', 'CNN'], fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Collect Predictions
# ══════════════════════════════════════════════════════════════════════════════

def collect_predictions(model, data_loader, device):
    model.eval()

    all_images = []
    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            all_images.append(images.cpu().numpy().squeeze(1))
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    return (np.concatenate(all_images),
            np.concatenate(all_labels),
            np.concatenate(all_preds))


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(model, data_loader, device, classes,
                          model_name="Model",
                          save_path="plots/confusion_matrix.png"):
    
    _, all_labels, all_preds = collect_predictions(model, data_loader, device)
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)

    ax.set_title(f'Confusion Matrix -- {model_name}',
                 fontsize=13, fontweight='bold', pad=15)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Failure Analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_failure_analysis(nn_model, cnn_model, data_loader, device,
                          classes, n_examples=10,
                          save_path="plots/failure_analysis.png"):
    """
    Finds images where NN is wrong but CNN is correct.
    """
    all_images, all_labels, nn_preds  = collect_predictions(nn_model,  data_loader, device)
    _,          _,          cnn_preds = collect_predictions(cnn_model, data_loader, device)

    failure_mask = (all_labels != nn_preds) & (all_labels == cnn_preds)
    failure_indices = np.where(failure_mask)[0]

    if len(failure_indices) == 0:
        print("No failure cases found where NN was wrong and CNN was right.")
        return

    n_show = min(n_examples, len(failure_indices))
    selected = np.random.choice(failure_indices, size=n_show, replace=False)

    n_cols = 5
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 3.2))

    for i, ax in enumerate(axes.flat):
        if i < len(selected):
            idx = selected[i]
            ax.imshow(all_images[idx], cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest')

            true_name = classes[all_labels[idx]]
            nn_name   = classes[nn_preds[idx]]
            cnn_name  = classes[cnn_preds[idx]]

            ax.set_title(
                f"True:  {true_name}\n"
                f"NN:    {nn_name} ✗\n"
                f"CNN:   {cnn_name} ✓",
                fontsize=7.5,
                color='#c0392b',
                loc='center'
            )
        ax.axis('off')

    plt.suptitle(
        f'Failure Analysis: NN Wrong — CNN Correct ({n_show} Examples)',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(nn_model, cnn_model, nn_history, cnn_history,
                        nn_acc, cnn_acc, test_loader, device, classes):
    
    print("\nGenerating Plots...")

    plot_training_curves(nn_history, cnn_history)

    plot_accuracy_comparison(nn_acc, cnn_acc)

    plot_confusion_matrix(nn_model,  test_loader, device, classes,
                          model_name="NN",
                          save_path="plots/confusion_matrix_nn.png")

    plot_confusion_matrix(cnn_model, test_loader, device, classes,
                          model_name="CNN",
                          save_path="plots/confusion_matrix_cnn.png")

    plot_failure_analysis(nn_model, cnn_model, test_loader,
                          device, classes, n_examples=10)

    print("\nAll plots saved in the parent directory's plots/ folder.")