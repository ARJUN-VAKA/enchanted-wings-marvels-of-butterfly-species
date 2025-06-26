import os
import shutil
import pandas as pd

# Define the base directory
data_dir = os.path.join(os.getcwd(), 'butterfly_data')
train_csv = os.path.join(data_dir, 'Training_set.csv')
train_dir = os.path.join(data_dir, 'train')
train_sorted_dir = os.path.join(data_dir, 'train_sorted')

# Create sorted directory if it doesn't exist
if os.path.exists(train_sorted_dir):
    shutil.rmtree(train_sorted_dir)
os.makedirs(train_sorted_dir, exist_ok=True)

# Load training data
train_df = pd.read_csv(train_csv)

# Check if train directory contains images
if not os.path.exists(train_dir):
    print(f"Error: train folder not found at {train_dir}")
else:
    train_images = {f.lower() for f in os.listdir(train_dir) if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])}
    print(f"Found {len(train_images)} images in {train_dir} with extensions .jpg, .jpeg, or .png")

    # Move images to species subfolders, handling case sensitivity
    missing_count = 0
    for index, row in train_df.iterrows():
        filename = row['filename'].lower()
        src = os.path.join(train_dir, next((f for f in os.listdir(train_dir) if f.lower() == filename), None))
        if src and os.path.exists(src):
            dst_dir = os.path.join(train_sorted_dir, row['label'])
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.move(src, dst)
        else:
            print(f"Image not found: {os.path.join(train_dir, row['filename'])}")
            missing_count += 1

    print(f"Images organized into {train_sorted_dir}")
    print(f"Number of missing images: {missing_count}")