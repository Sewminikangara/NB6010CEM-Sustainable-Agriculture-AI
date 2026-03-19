import torch
import torch.nn as nn
import timm
import logging

def get_model(num_classes, pretrained=False):
    # Defaulting to pretrained=False to avoid network hang during interactive sessions
    model_name = 'vit_tiny_patch16_224'
    
    try:
        model = timm.create_model(model_name, pretrained=pretrained)
        logging.info(f"Loaded {model_name} with pretrained={pretrained}")
    except Exception as e:
        logging.error(f"Failed to load {model_name}: {e}. Falling back to random initialization.")
        model = timm.create_model(model_name, pretrained=False)
    
    # Replace the classification head
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model
