import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ============================
# Step 1: Load Metadata
# ============================
metadata_path = 'E:/UMIT/MINI_project/implementation/HAM10000_metadata.csv'
image_folder = 'E:/UMIT/MINI_project/implementation/HAM10000_images'

metadata = pd.read_csv(metadata_path)
available_images = set(os.listdir(image_folder))
metadata['image_path'] = metadata['image_id'].apply(
    lambda x: os.path.join(image_folder, f"{x}.jpg") if f"{x}.jpg" in available_images else None
)
metadata.dropna(subset=['image_path'], inplace=True)

label_encoder = LabelEncoder()
metadata['label'] = label_encoder.fit_transform(metadata['dx'])
np.save("E:/UMIT/MINI_project/implementation/labels.npy", label_encoder.classes_)

print(f"✅ Total valid images found: {len(metadata)}")

# ============================
# Step 2: Load Images
# ============================
def load_images(metadata):
    images, labels = [], []
    skipped_files = 0
    for idx, row in metadata.iterrows():
        img_path = row['image_path']
        label = row['label']
        if not os.path.exists(img_path):
            skipped_files += 1
            continue
        image = cv2.imread(img_path)
        if image is None:
            skipped_files += 1
            continue
        image = cv2.resize(image, (224, 224)) / 255.0
        images.append(image)
        labels.append(label)
    print(f"✅ Successfully loaded {len(images)} images.")
    print(f"❗ Skipped {skipped_files} files (missing or unreadable).")
    return np.array(images), np.array(labels)

images, labels = load_images(metadata)

# ============================
# Step 3: Stratified Split
# ============================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(images, labels):
    X_train, X_test = images[train_idx], images[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

print(f"✅ Training Set: {len(X_train)} images")
print(f"✅ Testing Set: {len(X_test)} images")

# ============================
# Step 4: Data Augmentation
# ============================
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
datagen.fit(X_train)
print("✅ Data augmentation applied successfully.")

# ============================
# Step 5: Build the CNN Model
# ============================
model = keras.models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(labels)), activation='softmax')
])

# ============================
# Step 6: Compile the Model
# ============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================
# Step 7: Compute Class Weights
# ============================
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("✅ Class weights computed to handle imbalance.")

# ============================
# Step 8: Train the Model
# ============================
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# ============================
# Step 9: Evaluate the Model
# ============================
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# ============================
# Step 10: Confusion Matrix
# ============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ============================
# Step 11: Save the Model
# ============================
model.save('skinmantra_model.keras')
print("✅ Model saved successfully.")
