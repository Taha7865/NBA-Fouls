import os
import PIL
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Install the necessary libraries
# pip install tensorflow keras opencv-python scikit-learn
# For random images
# def load_data(data_dir, image_size, num_samples_per_class):
#     images = []
#     labels = []
#     for label in ['foul', 'non-foul']:
#         for _ in range(num_samples_per_class):
#             img = np.random.randint(0, 256, (image_size[0], image_size[1], 3), dtype=np.uint8)
#             images.append(img)
#             labels.append(label)
#     return np.array(images), np.array(labels)

def load_data(data_dir, image_size):
    images = []
    labels = []
    for label in ['foul', 'non-foul']:
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                img = PIL.Image.open(file_path)
                img = img.resize(image_size)
                img = np.array(img)
                images.append(img)
                labels.append(0 if label == 'foul' else 1)
    return np.array(images), np.array(labels)


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def predict_foul(model, image_path, image_size):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, image_size)
    img_expanded = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expanded)
    if prediction > 0.5:
        return 'non-foul'
    else:
        return 'foul'

if __name__ == "__main__":
    data_dir = 'C:\\Users\\tahaa\\OneDrive\\Desktop\\NBA AI MODEL\\basketball_foul_dataset'
    image_size = (224, 224)
# Using mock data
# num_samples_per_class = 100  # Set the number of samples per class as desired
# images, labels = load_data(data_dir, image_size, num_samples_per_class)

    # Load and preprocess data
    images, labels = load_data(data_dir, image_size)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_model(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save('foul_detection_model.h5')

    # Test the model on a new image
    test_image_path = 'C:/Users/tahaa/OneDrive/Desktop/NBA AI MODEL/test/test 1.jpg'
    prediction = predict_foul(model, test_image_path, image_size)
    print(f"The predicted label for the test image is: {prediction}")
