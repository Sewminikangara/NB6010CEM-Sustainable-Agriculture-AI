import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import config


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            img_list = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if max_per_class:
                img_list = img_list[:max_per_class]

            for img_name in img_list:
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(root_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, max_per_class=None):
    from src.augmentation import get_train_transforms, get_val_test_transforms

    if not os.path.exists(root_dir):
        print(f"Warning: Dataset directory {root_dir} not found.")
        return None, None, None

    train_dataset = PlantDiseaseDataset(root_dir, transform=get_train_transforms(), max_per_class=max_per_class)
    val_test_dataset = PlantDiseaseDataset(root_dir, transform=get_val_test_transforms(), max_per_class=max_per_class)

    indices = list(range(len(train_dataset)))
    train_indices, temp_indices = train_test_split(
        indices, test_size=(1 - config.TRAIN_SPLIT), stratify=train_dataset.labels, random_state=42
    )

    val_test_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(1 - val_test_ratio),
        stratify=[train_dataset.labels[i] for i in temp_indices], random_state=42
    )

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(val_test_dataset, val_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(val_test_dataset, test_indices), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
