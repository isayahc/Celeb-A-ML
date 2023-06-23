import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
import os
import random

# Create an argument parser
parser = argparse.ArgumentParser(description='Predict attributes using a trained model.')
parser.add_argument('image_path', type=str, nargs='?', default=None, help='The path to the image.')

# Parse the arguments
args = parser.parse_args()

# Define the model architecture
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(218, 178, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(40, activation='sigmoid'),
])

# Load the weights into the model
model.load_weights('./weights/model_weights.h5')

# If no image path is provided, select a random image from the dataset
if args.image_path is None:
    image_dir = '/home/isayahc/projects/machine_learning/celeb_face_training/datasets/celeba/img_align_celeba/img_align_celeba'
    args.image_path = random.choice([os.path.join(image_dir, f) for f in os.listdir(image_dir)])

# Load the image
img = load_img(args.image_path, target_size=(218, 178))

# Convert the image to an array and add a batch dimension
img = img_to_array(img)
img = np.expand_dims(img, axis=0)

# Scale the image
img = img / 255.

# Make a prediction
prediction = model.predict(img)

# List of CelebA attributes
attributes = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 
    'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 
    'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 
    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
    'Wearing_Necktie', 'Young'
]

# Convert the prediction to a binary decision
binary_prediction = np.where(prediction[0] > 0.5, 1, 0)

# Print the attributes that are predicted to be present
for i, is_present in enumerate(binary_prediction):
    if is_present:
        print(attributes[i])



# Print the prediction
print(prediction)
