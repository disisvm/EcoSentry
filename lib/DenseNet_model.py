import os
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from pca import load_and_preprocess_data, apply_pca

# Set your parent directory
parent_directory = '/kaggle/input/danger-of-extinction-animal-image-set/Danger Of Extinction'

# Load and preprocess data
image_data, labels, class_names = load_and_preprocess_data(parent_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Apply PCA for feature selection
X_train_pca, pca_components = apply_pca(X_train, n_components=10)

# Use DenseNet to extract features
base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
X_train_features = base_model.predict(X_train)

# Apply PCA for feature selection
n_components = 10
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))

# DenseNet Model
def build_densenet_model(input_shape, num_classes):
    base_model = DenseNet121(include_top=False, input_shape=input_shape, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Define input shape and number of classes
input_shape = (224, 224, 3)
num_classes = len(class_names)

# Build the model
model = build_densenet_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print("F1 Score:", f1_score(y_test, y_pred_classes, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred_classes))

model.save('DenseNet_model.h5')