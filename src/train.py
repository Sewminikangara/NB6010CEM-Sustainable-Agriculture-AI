"""
Training pipeline for MobileNetV2 and ViT models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import argparse

import config
from src.dataset import get_dataloaders
from src.models import mobilenet_v2, vit


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu', patience=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if dataloaders[phase] is None:
                continue

            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f"--> Saved best model (Acc: {best_acc:.4f})")

        if epochs_no_improve >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch}.')
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mobilenet', help='mobilenet or vit')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--subset', type=int, default=None, help='Max images per class (for quick runs)')
    args = parser.parse_args()

    train_loader, val_loader, _ = get_dataloaders(max_per_class=args.subset)
    dataloaders = {'train': train_loader, 'val': val_loader}

    if args.model == 'mobilenet':
        model = mobilenet_v2.get_model(config.NUM_CLASSES).to(config.DEVICE)
    else:
        model = vit.get_model(config.NUM_CLASSES).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler,
                        num_epochs=args.epochs, device=config.DEVICE, patience=5)

    save_path = os.path.join(config.MODEL_SAVE_DIR, f"{args.model}_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
