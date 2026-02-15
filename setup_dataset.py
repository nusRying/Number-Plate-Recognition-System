import zipfile
import os
import shutil
import random
from glob import glob

DATASET_DIR = "data/dataset"
PROCESSED_DIR = "data/yolo_data"

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def organize_dataset():
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(PROCESSED_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, split, 'labels'), exist_ok=True)

    # Extract all zips to a temp folder
    temp_dir = os.path.join(DATASET_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Extract labels
    extract_zip(os.path.join(DATASET_DIR, "labels.zip"), temp_dir)
    
    # Extract images (using just the first pack for speed, or all if needed)
    # Let's extract all image zips found
    image_zips = glob(os.path.join(DATASET_DIR, "VehiclesNepal*.zip"))
    for zip_file in image_zips:
        extract_zip(zip_file, temp_dir)

    # Now verify extraction and move files
    # Assuming labels are .txt and images are .jpg/.png in temp_dir or subfolders
    # We need to find where they extracted to.
    
    all_files = glob(os.path.join(temp_dir, "**/*.*"), recursive=True)
    images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    labels = [f for f in all_files if f.lower().endswith('.txt')]

    print(f"Found {len(images)} images and {len(labels)} labels.")

    # Match images with labels
    data_pairs = []
    for img_path in images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Look for corresponding label text file
        # Check if label exists in the labels list
        # This simple check assumes labels are in the same folder or flattened
        # Let's try to find the label file in the labels list
        label_path = next((l for l in labels if os.path.splitext(os.path.basename(l))[0] == base_name), None)
        
        if label_path:
            data_pairs.append((img_path, label_path))

    print(f"Matched {len(data_pairs)} image-label pairs.")

    # Shuffle and split
    random.shuffle(data_pairs)
    split_idx = int(len(data_pairs) * 0.8)
    train_pairs = data_pairs[:split_idx]
    val_pairs = data_pairs[split_idx:]

    def move_files(pairs, split):
        for img, lbl in pairs:
            shutil.copy(img, os.path.join(PROCESSED_DIR, split, 'images', os.path.basename(img)))
            shutil.copy(lbl, os.path.join(PROCESSED_DIR, split, 'labels', os.path.basename(lbl)))

    move_files(train_pairs, 'train')
    move_files(val_pairs, 'val')

    # Cleanup temp
    shutil.rmtree(temp_dir)
    
    # Create data.yaml
    yaml_content = f"""
path: ./yolo_data  # dataset root dir (relative to where script is run)
train: train/images
val: val/images

# Classes
names:
  0: License Plate
"""
    with open(os.path.join("data", "data.yaml"), "w") as f:
        f.write(yaml_content.strip())
    
    print("Dataset setup complete.")

if __name__ == "__main__":
    organize_dataset()
