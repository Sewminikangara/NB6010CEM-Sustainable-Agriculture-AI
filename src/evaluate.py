# Model Evaluation and Metric Visualization
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os

import config
from src.dataset import get_dataloaders
from src.models import mobilenet_v2, vit

def evaluate_model(model, test_loader, device, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {acc:.4f}')
    
    report = classification_report(all_labels, all_preds, target_names=test_loader.dataset.dataset.classes)
    print("Classification Report:")
    print(report)
    
    # Save report
    report_path = os.path.join(config.LOG_SAVE_DIR, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\nOverall Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_loader.dataset.dataset.classes, yticklabels=test_loader.dataset.dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    
    cm_path = os.path.join(config.PLOT_SAVE_DIR, f"{model_name}_cm.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mobilenet', help='mobilenet or vit')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    args = parser.parse_args()

    _, _, test_loader = get_dataloaders()
    
    if args.model == 'mobilenet':
        model = mobilenet_v2.get_model(config.NUM_CLASSES).to(config.DEVICE)
    else:
        model = vit.get_model(config.NUM_CLASSES).to(config.DEVICE)

    model.load_state_dict(torch.load(args.weights, map_location=config.DEVICE))
    evaluate_model(model, test_loader, config.DEVICE, args.model)
