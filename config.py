import os
import torch

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "PlantVillage_Clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_SAVE_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_SAVE_DIR = os.path.join(OUTPUT_DIR, "logs")

for d in [OUTPUT_DIR, MODEL_SAVE_DIR, PLOT_SAVE_DIR, LOG_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Dataset ---
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 38

# --- Training ---
EPOCHS = 25
LEARNING_RATE = 1e-4

# --- Prediction ---
CONFIDENCE_THRESHOLD = 0.15
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- LLM ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
