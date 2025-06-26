import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import numpy as np

img_size = (224, 224)
batch_size = 32
num_classes = 75  # Matches unique species count from Training_set.csv

data_dir = os.path.join(os.getcwd(), 'butterfly_data')
train_sorted_dir = os.path.join(data_dir, 'train_sorted')
test_dir = os.path.join(data_dir, 'test')
test_csv = os.path.join(data_dir, 'Testing_set.csv')

# Load test data CSV
test_df = pd.read_csv(test_csv)
test_df['image_path'] = test_df['filename'].apply(lambda x: os.path.join(test_dir, x))

# Validate image paths
valid_paths = []
for path in test_df['image_path']:
    if os.path.exists(path):
        valid_paths.append(path)
    else:
        print(f"Warning: Image not found at {path}")
test_df = test_df[test_df['image_path'].isin(valid_paths)].reset_index(drop=True)

# Data generators with validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    train_sorted_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Use training subset
)

val_generator = train_datagen.flow_from_directory(
    train_sorted_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Use validation subset
)

test_datagen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col=None,
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# Model definition
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Evaluate with error handling
try:
    loss, accuracy = model.evaluate(test_datagen)
    print(f'Test Accuracy: {accuracy:.2f}')
except ValueError as e:
    print(f"Error during evaluation: {e}. Check test data integrity.")

os.makedirs('model', exist_ok=True)
model.save('model/butterfly_classifier.h5')
print("Model saved to model/butterfly_classifier.h5")

# Predict and map to species names
if len(test_df) > 0:
    test_datagen.reset()
    predictions = model.predict(test_datagen)
    predicted_classes = predictions.argmax(axis=1)
    class_labels = list(train_generator.class_indices.keys())  # Species names from train_sorted
    predicted_species = [class_labels[idx] for idx in predicted_classes]

    # Match predictions with image names
    test_files = [os.path.basename(f) for f in test_datagen.filenames]
    results_df = pd.DataFrame({'image_name': test_files, 'predicted_species': predicted_species})
    results_df.to_csv('test_predictions.csv', index=False)
    print("\nSample Predictions:")
    print(results_df.head())
else:
    print("No valid test images found. Predictions not generated.")