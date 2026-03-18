"""
Inference pipeline for plant disease classification.
"""
import torch
from torchvision import transforms
from PIL import Image
import os

import config
from src.models import mobilenet_v2, vit
from src.llm_agent import PlantAdvisor


class DiseasePredictor:
    def __init__(self, model_type='mobilenet', weights_path=None):
        self.device = config.DEVICE
        self.model_type = model_type

        if model_type == 'mobilenet':
            self.model = mobilenet_v2.get_model(config.NUM_CLASSES, pretrained=False).to(self.device)
        else:
            self.model = vit.get_model(config.NUM_CLASSES, pretrained=False).to(self.device)

        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.classes = sorted(os.listdir(config.DATA_DIR))
        self.advisor = PlantAdvisor()

    @staticmethod
    def format_class_name(name):
        """Convert raw class label (e.g. Tomato___Early_blight) to readable format."""
        if "___" in name:
            plant, disease = name.split("___", 1)
            if disease.lower().startswith(plant.lower()):
                disease = disease[len(plant):].lstrip("_")
            plant_fmt = plant.replace("_", " ").title()
            disease_fmt = disease.replace("_", " ").title()
            return f"{plant_fmt} - {disease_fmt}"
        return name.replace("_", " ").title()

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        confidence_val = confidence.item()
        raw_class_name = self.classes[predicted_idx.item()]
        is_known = confidence_val >= config.CONFIDENCE_THRESHOLD

        if is_known:
            advice = self.advisor.get_advice(raw_class_name)
            display_name = self.format_class_name(raw_class_name)
        else:
            display_name = "Unknown / Non-Plant"
            advice = (
                "### Low Confidence Prediction\n\n"
                "The system could not confidently identify a plant disease in this image.\n\n"
                "**Possible reasons:**\n"
                "- The image does not contain a plant leaf.\n"
                "- Lighting or image quality is insufficient.\n"
                "- The disease is not in our database (38 classes).\n\n"
                "*Please upload a clear, well-lit photograph of a single leaf.*"
            )

        return {
            "class": display_name,
            "raw_class": raw_class_name,
            "confidence": confidence_val,
            "advice": advice,
            "is_known": is_known
        }
