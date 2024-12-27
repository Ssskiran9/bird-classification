import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import math

import numpy as np
import cv2
import os
import shutil
import PIL
from sklearn.model_selection import train_test_split

# Define model architecture
def create_model(input_shape=(224, 224, 3), num_classes=6):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


# Create a TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

image_shape = (224, 224, 3)

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1.0/255)

import os
import shutil
from sklearn.model_selection import train_test_split

# Paths for dataset splits
dataset_dir = "C:\\Users\\saiki\\Downloads\\archive (4)\\Bird Speciees Dataset"  # Original dataset path
output_dir = "C:\\Users\\saiki\\Downloads\\archive (4)\\Bird Speciees Dataset_Split"  # Directory for split data
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Create directories for train, val, test
splits = ["train", "test"]
for split in splits:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Ensure correct splits
if train_ratio + val_ratio + test_ratio != 1:
    raise ValueError("Train, validation, and test ratios must sum to 1.")

# Split each class
for cls in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(class_path):
        continue
    


    for f in os.listdir(class_path):
        try:
            img = PIL.Image.open(os.path.join(class_path, f))
            img.verify()
        except (IOError, SyntaxError):
            print(f"Corrupted file: {f}")

    # Create class subdirectories in each split
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

        # Get all files for the class
        files = os.listdir(class_path)
        files = [os.path.join(class_path, f) for f in files if os.path.isfile(os.path.join(class_path, f))]

        # Split the data into train, validation, and test
    train_files, test_files = train_test_split(files, test_size= test_ratio, random_state=42)
    



   # Copy the files to the respective folders
    for f in train_files:
        shutil.copy(f, os.path.join(output_dir, "train", cls))
    #for f in val_files:
    #    shutil.copy(f, os.path.join(output_dir, "val", cls))
    for f in test_files:
        shutil.copy(f, os.path.join(output_dir, "test", cls))

print("Data has been split into train validation qand test sets.")















batch_size = 16

# Train Image Generator for multi-class classification
train_image_gen = train_datagen.flow_from_directory(os.path.join(output_dir, "train"),
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='categorical'  # For multi-class
                                                )

test_image_gen = train_datagen.flow_from_directory(os.path.join(output_dir, "test"),
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='categorical')



train_image_gen.class_indices

# Calculate steps_per_epoch and validation_steps
#train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
#val_steps = math.ceil(val_generator.samples / val_generator.batch_size)


# Create and train model
model = create_model(num_classes=len(train_image_gen.class_indices))
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(
    train_image_gen,
    epochs=10,
    callbacks = [tensorboard_callback],
    #steps_per_epoch= train_steps,
    #validation_steps=val_steps,
    verbose=1
)

model.summary()



# Evaluate model
test_loss, test_accuracy = model.evaluate(test_image_gen)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save model
model.save("bird_cnn_model.keras")

# Real-time camera prediction
def predict_bird_species():
    cap = cv2.VideoCapture(0)
    class_names = list(train_image_gen.class_indices.keys())
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        resized = cv2.resize(frame, (224, 224))
        normalized = resized / 255.0
        batch = np.expand_dims(normalized, axis=0)
        
        # Make prediction
        predictions = model.predict(batch)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        # Display result
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Bird Species Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Start real-time prediction
predict_bird_species()