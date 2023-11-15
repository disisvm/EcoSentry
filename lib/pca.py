import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Import image enhancement function
import image_enhancements as enhancement

# Function to load and preprocess data
def load_and_preprocess_data(parent_dir, target_size=(224, 224)):
    labels = []
    image_data = []

    for animal_dir in os.listdir(parent_dir):
        animal_path = os.path.join(parent_dir, animal_dir)
        if os.path.isdir(animal_path):
            for image_file in os.listdir(animal_path):
                image_path = os.path.join(animal_path, image_file)
                img = load_img(image_path, target_size=target_size)
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                image_data.append(img_array)
                labels.append(animal_dir)

    # Convert labels to numeric
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    return np.array(image_data), encoded_labels, label_encoder.classes_

def visualize_data_distribution(labels, class_names):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels, palette='viridis')
    plt.title('Distribution of Animal Classes')
    plt.xlabel('Class')
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45)
    plt.show()

from sklearn.decomposition import PCA

def apply_pca(X_train, n_components=None):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
    return X_train_pca, pca.components_  # Return the components directly


'''# Set your parent directory
parent_directory = '../src/Danger Of Extinction'

# Load and preprocess data
image_data, labels, class_names = load_and_preprocess_data(parent_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Visualize data distribution
visualize_data_distribution(labels, class_names)

# Apply PCA for feature selection
X_train_pca, pca = apply_pca(X_train.reshape(X_train.shape[0], -1), n_components=10)
selected_features_pca = X_train.reshape(X_train.shape[0], -1)[:, pca.components_]

# Visualize PCA components
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Principal Components')
plt.show()'''

