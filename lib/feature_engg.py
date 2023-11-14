import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import mutual_info_score
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


def feature_selection(X_train, y_train, k_best=10):
    selector = SelectKBest(f_classif, k=k_best)
    X_train_selected = selector.fit_transform(X_train, y_train)
    return X_train_selected, selector.get_support(indices=True)


def feature_importance(X_train, y_train, feature_names):
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]
    selected_indices = indices[:10]  # Adjust the number of top features to show

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=importances[selected_indices], y=np.array(feature_names)[selected_indices])
    for i in selected_indices:
        print("Importance scores:", np.array(feature_names)[i], importances[i])
    ax.set_title('Top Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_xticklabels(['{:.4f}'.format(importance) for importance in importances[selected_indices]])
    plt.show()


def mutual_information(X_train, y_train, feature_names):
    mi_scores = np.zeros(X_train.shape[1])  # Initialize array to store mutual information scores

    for feature_idx in range(X_train.shape[1]):
        mi_scores[feature_idx] = mutual_info_score(X_train[:, feature_idx], y_train)

    selected_indices = np.argsort(mi_scores)[::-1][:10]

    # Plot the mutual information scores
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=mi_scores[selected_indices], y=np.array(feature_names)[selected_indices])
    for i in selected_indices:
        print("MI scores:", np.array(feature_names)[i], mi_scores[i])
    ax.set_title('Top Mutual Information Scores')
    ax.set_xlabel('Mutual Information Score')
    ax.set_ylabel('Feature')
    ax.set_xticklabels(['{:.4f}'.format(score) for score in mi_scores[selected_indices]])
    plt.show()


# Set your parent directory
parent_directory = '../src/Danger Of Extinction'

# Load and preprocess data
image_data, labels, class_names = load_and_preprocess_data(parent_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Visualize data distribution
visualize_data_distribution(labels, class_names)

# Feature Selection
X_train_selected, selected_indices = feature_selection(X_train.reshape(X_train.shape[0], -1), y_train)
selected_features = X_train.reshape(X_train.shape[0], -1)[:, selected_indices]

# Visualize feature importance's
feature_importance(selected_features, y_train, selected_indices)

# Visualize mutual information scores
mutual_information(selected_features, y_train, selected_indices)
