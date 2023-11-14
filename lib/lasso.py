import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Lasso
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


def lasso_feature_selection(X_train, y_train, alpha=0.01):
    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    selected_indices = np.where(model.coef_ != 0)[0]

    return selected_indices


def visualize_lasso_coefficients(coef, feature_names, class_names):
    if len(coef) != len(feature_names):
        raise ValueError("Lengths of coef and feature_names must be the same")

    # Plot the LASSO coefficients for each class
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=coef, y=feature_names, hue=class_names, dodge=True)
    ax.set_title('LASSO Coefficients for Each Class')
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Feature')
    ax.legend(title='Class')
    ax.set_xticklabels(['{:.4f}'.format(c) for c in coef])
    plt.show()




# Set your parent directory
parent_directory = '../src/Danger Of Extinction'

# Load and preprocess data
image_data, labels, class_names = load_and_preprocess_data(parent_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Visualize data distribution
visualize_data_distribution(labels, class_names)

# LASSO Feature Selection
lasso_selected_indices = lasso_feature_selection(X_train.reshape(X_train.shape[0], -1), y_train)
lasso_selected_features = X_train.reshape(X_train.shape[0], -1)[:, lasso_selected_indices]

# Visualize LASSO coefficients
visualize_lasso_coefficients(lasso_selected_indices, class_names)
