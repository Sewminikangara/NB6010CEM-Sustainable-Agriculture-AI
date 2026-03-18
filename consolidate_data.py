import os
import shutil
import re

# Target class names 
target_classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

source_root = "data/PlantVillage"
dest_root = "data/PlantVillage_Clean"

if os.path.exists(dest_root):
    shutil.rmtree(dest_root)
os.makedirs(dest_root)

def normalize_name(name):
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

# Map normalized names to target names
mapping = {normalize_name(cls): cls for cls in target_classes}

# Special manual overrides for tricky names
mapping[normalize_name("Pepper__bell___Bacterial_spot")] = "Pepper,_bell___Bacterial_spot"
mapping[normalize_name("Pepper__bell___healthy")] = "Pepper,_bell___healthy"
mapping[normalize_name("Corn__maize___Cercospora_leaf_spot_Gray_leaf_spot")] = "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot"
mapping[normalize_name("Tomato__Tomato_YellowLeaf__Curl_Virus")] = "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
mapping[normalize_name("Tomato_Spider_mites_Two_spotted_spider_mite")] = "Tomato___Spider_mites_Two-spotted_spider_mite"

print("Starting consolidation...")

found_sources = {}

for root, dirs, files in os.walk(source_root):
    if not files: continue
    
    # Check if this directory has images
    has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
    if not has_images: continue
    
    dir_name = os.path.basename(root)
    norm_name = normalize_name(dir_name)
    
    if norm_name in mapping:
        target_name = mapping[norm_name]
        if target_name not in found_sources:
            found_sources[target_name] = []
        found_sources[target_name].append(root)

# Copy files
for target_name, sources in found_sources.items():
    dest_dir = os.path.join(dest_root, target_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    cnt = 0
    for src_dir in sources:
        for f in os.listdir(src_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(src_dir, f)
                dest_file = os.path.join(dest_dir, f"{cnt}_{f}")
                shutil.copy2(src_file, dest_file)
                cnt += 1
    print(f"Copied {cnt} images for {target_name}")

# Check missing
missing = set(target_classes) - set(found_sources.keys())
if missing:
    print(f"\nWARNING: Missing {len(missing)} classes: {missing}")
else:
    print("\nSUCCESS: All 38 classes processed.")
