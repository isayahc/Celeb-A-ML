# File: train_model.py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Define the path to the CelebA dataset
data_dir = 'datasets/celeba/'

# Define the path for saving weights
weights_dir = './weights'


if not os.path.exists(data_dir):
    raise FileNotFoundError("Please run setup_dataset.py or setup_dataset.sh")

# Create the directory if it doesn't exist
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)


# Load the attribute data
attr_df = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.csv'))

# Convert to binary (1 for positive, 0 for negative)
attr_df.replace(to_replace=-1, value=0, inplace=True)

# Split the data into train and validation sets
train_df, val_df = train_test_split(attr_df, test_size=0.2, random_state=42)

# Define data generators for training and validation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)


train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=os.path.join(data_dir, 'img_align_celeba/img_align_celeba'),
    x_col="image_id",
    y_col=list(train_df.columns[1:]),
    class_mode="raw",
    target_size=(218, 178),
    batch_size=32,
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=os.path.join(data_dir, 'img_align_celeba/img_align_celeba'),
    x_col="image_id",
    y_col=list(val_df.columns[1:]),
    class_mode="raw",
    target_size=(218, 178),
    batch_size=32,
)

# Define the model
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

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1,
    callbacks=[early_stopping],
)

# Save the model weights
model.save_weights(os.path.join(weights_dir, 'model_weights.h5'))

