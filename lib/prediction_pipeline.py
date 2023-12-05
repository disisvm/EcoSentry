import numpy as np
from tensorflow.keras.models import load_model

from image_enhancements import enhance_image
from email_trigger import send_email

# Load the pre-trained models
vgg_model = load_model('../models/VGG_model_final.h5')
mobilenet_model = load_model('../models/MobileNetV2_model.h5')
densenet_model = load_model('../models/DenseNet_model.h5')
efficientnet_model = load_model('../models/EfficientNet_model.h5')


def ensemble_predict(img_path):
    print(img_path)

    img = enhance_image(img_path, (224, 224))

    vgg_pred = vgg_model.predict(img)
    mobilenet_pred = mobilenet_model.predict(img)

    # Check if there is an animal in the image using VGG and MobileNet predictions
    if (vgg_pred[0][0] + mobilenet_pred[0][0]) / 2 > 0.5:
        densenet_pred = densenet_model.predict(img)
        efficientnet_pred = efficientnet_model.predict(img)

        # Check if the prediction confidence for species identification is above the threshold
        if max(densenet_pred[0]) > 0.8 or max(efficientnet_pred[0]) > 0.8:
            species_index = np.argmax(densenet_pred[0]) if max(densenet_pred[0]) > 0.8 else np.argmax(
                efficientnet_pred[0])

            class_labels = ['Chimpanzee', 'Panthers', 'cheetahs', 'Lion', 'Amur_Leopard', 'Orangutan', 'Panda',
                            'Jaguars', 'Rhino', 'Arctic_Fox', 'African_Elephant']

            return {
                'filename': img_path,  # Replace with the actual filename
                'class': {class_labels[species_index]},  # Replace with the actual class identified
                'confidence': max(densenet_pred[0], efficientnet_pred[0])  # Replace with the actual confidence score
            }
        else:
            return {
                'filename': img_path,
                'class': 'No animal',
                'confidence': 1 - (max(densenet_pred[0], efficientnet_pred[0]))
            }
    else:
        return {
            'filename': img_path,
            'class': 'No animal',
            'confidence': (vgg_pred[0][0] + mobilenet_pred[0][0]) / 2
        }
