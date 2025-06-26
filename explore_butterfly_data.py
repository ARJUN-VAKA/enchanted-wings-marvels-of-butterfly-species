import pandas as pd
import os

# Define the base directory
data_dir = os.path.join(os.getcwd(), 'butterfly_data')

# Paths to CSV files and folders
train_csv = os.path.join(data_dir, 'Training_set.csv')
test_csv = os.path.join(data_dir, 'Testing_set.csv')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Debug: Print paths to verify
print(f"Looking for Training_set.csv at: {train_csv}")
print(f"Looking for Testing_set.csv at: {test_csv}")
print(f"Looking for train folder at: {train_dir}")
print(f"Looking for test folder at: {test_dir}")

# Check if files and folders exist
if not os.path.exists(train_csv):
    print(f"Error: Training_set.csv not found at {train_csv}")
if not os.path.exists(test_csv):
    print(f"Error: Testing_set.csv not found at {test_csv}")
if not os.path.exists(train_dir):
    print(f"Error: train folder not found at {train_dir}")
if not os.path.exists(test_dir):
    print(f"Error: test folder not found at {test_dir}")

# Load CSV files if they exist
if os.path.exists(train_csv) and os.path.exists(test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Explore training data
    print("\nTraining Data Overview:")
    print(f"Number of training images: {len(train_df)}")
    print(f"Unique species: {train_df['label'].nunique()}")
    print("Sample of training data:")
    print(train_df.head())

    # Explore test data
    print("\nTest Data Overview:")
    print(f"Number of test images: {len(test_df)}")
    print("Sample of test data:")
    print(test_df.head())

    # Check image files with proper extension handling
    train_images = {f.lower() for f in os.listdir(train_dir) if f.lower().endswith('.jpg')}
    test_images = {f.lower() for f in os.listdir(test_dir) if f.lower().endswith('.jpg')}
    train_csv_images = {f.lower() for f in train_df['filename']}
    test_csv_images = {f.lower() for f in test_df['filename']}

    print(f"\nNumber of images in train folder: {len(train_images)}")
    print(f"Number of images in test folder: {len(test_images)}")
    print(f"Images in train CSV but not in folder: {train_csv_images - train_images}")
    print(f"Images in test CSV but not in folder: {test_csv_images - test_images}")
else:
    print("Please ensure all files and folders are in C:\\butterfly model\\butterfly_data.")