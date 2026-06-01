"""
OpenCLIP ViT-H/14 Loader
--------------------------
Loads OpenCLIP ViT-H/14 pretrained on LAION-2B.
Zero-shot model — no training required.
Provides model and dataloaders ready for embedding extraction.

Model      : ViT-H/14
Pretrained : LAION-2B (2 billion image-text pairs)
Embedding  : 1024 dimensions
"""

import os
import torch
import open_clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# CONFIG 
DATA_DIR       = '/your/path/coco'
MODEL_NAME     = 'ViT-H-14'
PRETRAINED     = 'laion2b_s32b_b79k'
EMBEDDING_DIM  = 1024
BATCH_SIZE     = 8     # smaller batch due to large model size (632M params)
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
IGNORE_FOLDERS = {'.Trash-1001', 'lost+found'}

# Download model to local drive instead of home directory
os.environ['HF_HOME'] = '/your/path/coco/hf_cache'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# DATASET 
class FilteredImageFolder(ImageFolder):
    # Excludes system folders from dataset
    def find_classes(self, directory):
        classes = [
            f for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))
            and f not in IGNORE_FOLDERS
        ]
        classes.sort()
        return classes, {cls: idx for idx, cls in enumerate(classes)}


class SafeDataset(torch.utils.data.Dataset):
    # Skips corrupted images gracefully
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        try:
            return self.subset[idx]
        except Exception:
            return None


def collate_skip_none(batch):
    # Removes None entries caused by corrupted images
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# DATALOADERS 
def get_dataloaders(preprocess):
    # Uses CLIP's own preprocessing — handles resizing and normalisation
    dataset = FilteredImageFolder(
        root=DATA_DIR,
        transform=preprocess,
        is_valid_file=lambda p: os.path.splitext(p)[1].lower() in IMAGE_EXTS
    )

    # Stratified split: 70% train / 15% val / 15% test
    indices = list(range(len(dataset)))
    labels  = [dataset.targets[i] for i in indices]

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=42
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.17647,
        stratify=train_val_labels, random_state=42
    )

    print(f"Train : {len(train_idx)} | Val : {len(val_idx)} | Test : {len(test_idx)}")

    loader_args = dict(batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_skip_none,
                       num_workers=4, pin_memory=True)

    train_loader = DataLoader(SafeDataset(Subset(dataset, train_idx)), **loader_args)
    val_loader   = DataLoader(SafeDataset(Subset(dataset, val_idx)),   **loader_args)
    test_loader  = DataLoader(SafeDataset(Subset(dataset, test_idx)),  **loader_args)

    return train_loader, val_loader, test_loader


# LOAD MODEL 
# create_model_and_transforms returns model, train_preprocess, val_preprocess
# val_preprocess used for consistent embedding extraction (no augmentation)
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
model = model.to(device).eval()
print(f"OpenCLIP {MODEL_NAME} loaded (pretrained: {PRETRAINED})")

# LOAD DATALOADERS 
train_loader, val_loader, test_loader = get_dataloaders(preprocess)
