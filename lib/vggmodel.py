import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
import optuna
import shap


# Function to define the model architecture
def create_model(trial):
    animal_directory = '/kaggle/input/all-images-for-model/All Images Enhanced/All Images Enhanced'
    no_animal_directory = '/kaggle/input/no-animal-images/No animals enchanced/No animals enchanced'

    animal_files = [os.path.join(animal_directory, file) for file in os.listdir(animal_directory) if
                    file.endswith(('.jpg', '.jpeg', '.png'))]
    animal_labels = ['Animal'] * len(animal_files)

    no_animal_files = [os.path.join(no_animal_directory, file) for file in os.listdir(no_animal_directory) if
                       file.endswith(('.jpg', '.jpeg', '.png'))]
    no_animal_labels = ['No Animal'] * len(no_animal_files)

    df_animal = pd.DataFrame({'file_path': animal_files, 'label': animal_labels})
    df_no_animal = pd.DataFrame({'file_path': no_animal_files, 'label': no_animal_labels})
    df = pd.concat([df_animal, df_no_animal], ignore_index=True)

    datagen = ImageDataGenerator(rescale=1. / 255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 validation_split=0.2)

    image_size = (224, 224)
    batch_size = 32

    generator = datagen.flow_from_dataframe(
        df,
        x_col='file_path',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)

    # Use Optuna to search for optimal hyperparameters
    num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
    for i in range(num_dense_layers):
        num_units = trial.suggest_int(f'units_dense_{i}', 32, 512, log=True)
        x = Dense(num_units, activation='relu')(x)

    prediction = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=prediction)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Objective function for Optuna optimization
def objective(trial):
    model = create_model(trial)

    model.fit(
        generator,
        steps_per_epoch=len(generator),
        epochs=5,  # You can adjust the number of epochs based on your dataset
        verbose=0
    )

    evaluation = model.evaluate(generator)
    return evaluation[1]  # Return accuracy as the objective


# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=8)  # You can adjust the number of trials

# Get the best hyperparameters
best_params = study.best_params

# Train the final model with the best hyperparameters
final_model = create_model(study.best_trial)
final_model.fit(
    generator,
    steps_per_epoch=len(generator),
    epochs=20
)

# Evaluate the final model
evaluation = final_model.evaluate(generator)
print(f"Validation Loss: {evaluation[0]}")
print(f"Validation Accuracy: {evaluation[1]}")

# Save the final model
final_model.save("VGG_model_final.h5")