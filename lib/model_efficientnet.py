import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_and_preprocess_data(parent_dir, target_size=(224, 224)):
    image_data = []
    labels = []
    class_names = []

    for animal_name in os.listdir(parent_dir):
        animal_path = os.path.join(parent_dir, animal_name)
        if os.path.isdir(animal_path):
            class_names.append(animal_name)
            for image_file in os.listdir(animal_path):
                image_path = os.path.join(animal_path, image_file)
                img = load_img(image_path, target_size=target_size)
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                image_data.append(img_array)
                labels.append(class_names.index(animal_name))

    return np.array(image_data), np.array(labels), class_names


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
    return X_train_pca, pca.components_  # Return the components directly



import os
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import optuna
import shap

# Set your parent directory
parent_directory = '/kaggle/input/all-images-for-model/Danger Of Extinction Enchanced/Danger Of Extinction Enchanced'


# Load and preprocess data
image_data, labels, class_names = load_and_preprocess_data(parent_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Apply PCA for feature selection
n_components = 10
X_train_pca, pca_components = apply_pca(X_train, n_components=n_components)

# EfficientNet Model
def build_efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Define input shape and number of classes
input_shape = (224, 224, 3)
num_classes = len(class_names)

# Define the Optuna objective function
def objective(trial):
    # Build the model
    model = build_efficientnet_model(input_shape, num_classes)

    # Compile the model with Optuna suggested learning rate
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)

    # Evaluate the model on the validation set
    val_loss = history.history['val_loss'][-1]
    return val_loss

# Perform hyperparameter tuning with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)



# Get the best parameters
best_params = study.best_params

# Build the final model with the best hyperparameters
final_model = build_efficientnet_model(input_shape, num_classes)
final_model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print("F1 Score:", f1_score(y_test, y_pred_classes, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred_classes))

# Save the final model
final_model.save('EfficientNet_model.h5')

# Use SHAP for model interpretation
explainer = shap.Explainer(final_model)
shap_values = explainer.shap_values(X_test)

# Visualize SHAP values
shap.summary_plot(shap_values, X_test, feature_names=class_names)