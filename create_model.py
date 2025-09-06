# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Labels for our model
LABELS = {
    0: "แมวมีปุกบินไม่ได้",
    1: "หมาล่าอร่อยดี",
    2: "ยีราฟคอสั้นกินข้าว"
}

# Define the fixed image size for processing
IMAGE_SIZE = (64, 64)

def load_images_from_folder(folder_path, label_id):
    """
    Loads images from a specified folder, resizes them, converts to RGB,
    flattens pixel data, and assigns a label ID.
    """
    images_data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path)
                img = img.resize(IMAGE_SIZE) # Resize all images to a fixed size
                img = img.convert('RGB')     # Ensure 3 channels (Red, Green, Blue)
                
                # Flatten the image into a 1D array of pixel values
                img_array = np.array(img).flatten()
                images_data.append(img_array)
                labels.append(label_id)
            except Exception as e:
                print(f"Warning: Could not load/process image {img_path}: {e}")
    return images_data, labels

if __name__ == "__main__":
    data_dir = "image_data_animals" # New data directory name
    cat_dir = os.path.join(data_dir, "cat")
    dog_dir = os.path.join(data_dir, "dog")
    giraffe_dir = os.path.join(data_dir, "giraffe")

    # Create directories if they don't exist
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)
    os.makedirs(giraffe_dir, exist_ok=True)

    print(f"Please place your image files for cats in '{cat_dir}'")
    print(f"Please place your image files for dogs in '{dog_dir}'")
    print(f"Please place your image files for giraffes in '{giraffe_dir}'")
    print("Each folder should contain at least 20-30 diverse images for good training results.")
    print("-" * 50)

    # Load image data for each category
    print("Loading cat images...")
    cat_images, cat_labels = load_images_from_folder(cat_dir, 0)
    print(f"Loaded {len(cat_images)} cat images.")

    print("Loading dog images...")
    dog_images, dog_labels = load_images_from_folder(dog_dir, 1)
    print(f"Loaded {len(dog_images)} dog images.")

    print("Loading giraffe images...")
    giraffe_images, giraffe_labels = load_images_from_folder(giraffe_dir, 2)
    print(f"Loaded {len(giraffe_images)} giraffe images.")

    all_images = cat_images + dog_images + giraffe_images
    all_labels = cat_labels + dog_labels + giraffe_labels

    if not all_images:
        print("No images found in any of the directories. Please add images and run again.")
    else:
        X = np.array(all_images)
        y = np.array(all_labels)

        # Split data into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"\nTraining with {len(X_train)} images, testing with {len(X_test)} images.")

        # Create and train the SVM model
        print("Training SVM model... This might take a moment.")
        model = svm.SVC(kernel='linear', C=1.0, random_state=42) # Using a linear kernel for simplicity
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy on the test set: {accuracy:.2f}")

        # Save the trained model
        model_filename = 'img_models.sav' # New model filename
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Trained model saved successfully as '{model_filename}'.")