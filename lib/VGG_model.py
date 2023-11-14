from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mutual_info_score
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Import feature engineering functions
from feature_engg import load_and_preprocess_data, visualize_data_distribution, feature_selection, feature_importance, mutual_information

# re-size all the images to this
IMAGE_SIZE = [224, 224]

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

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(class_names), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()


def f1_score(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1


# tell the model what cost and optimization method to use
model.compile(
    loss='sparse_categorical_crossentropy',  # Changed to categorical crossentropy
    optimizer='adam',
    metrics=['accuracy', f1_score]
)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow(X_train, y_train, target_size=(224, 224), batch_size=32)
validation_set = test_datagen.flow(X_val, y_val, target_size=(224, 224), batch_size=32)
test_set = test_datagen.flow(X_test, y_test, target_size=(224, 224), batch_size=32)

# fit the model
r = model.fit_generator(
    training_set,
    validation_data=validation_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(validation_set)
)

# Save the model
model.save('animal_identification_model.h5')
