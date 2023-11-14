import os
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import mutual_info_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Import specific functions from feature_eng
from feature_engg import load_and_preprocess_data, feature_selection

# Set your parent directory
parent_directory = '../src/Danger Of Extinction'

# Load and preprocess data
image_data, labels, class_names = load_and_preprocess_data(parent_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Feature Selection
X_train_selected, selected_indices = feature_selection(X_train.reshape(X_train.shape[0], -1), y_train)
selected_features = X_train.reshape(X_train.shape[0], -1)[:, selected_indices]


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
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(selected_features, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
X_test_selected = X_test.reshape(X_test.shape[0], -1)[:, selected_indices]
y_pred = model.predict(X_test_selected)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print("F1 Score:", f1_score(y_test, y_pred_classes, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred_classes))

model.save('DenseNet_model.h5')
