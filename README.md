# Practical no.1 : Linear regression by using Deep Neural network : Implement Boston housing price Prediction problem by Linear regression using Deep Neural network. Use Boston House price prediction dataset.

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error

# Step 1: Load the Boston Housing dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Step 2: Standardize the data
sca = StandardScaler()
x_train = sca.fit_transform(x_train)  # Fit and transform on training data
x_test = sca.transform(x_test)  # Only transform test data

# Step 3: Build the model
model = models.Sequential()

# Add Dense layers to the model
model.add(layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)))  # First hidden layer
model.add(layers.Dense(64, activation='relu'))  # Second hidden layer
model.add(layers.Dense(1))  # Output layer for regression (single unit)

# Step 4: Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Mean Squared Error for regression

# Step 5: Train the model
history = model.fit(x_train, y_train, epochs=30, batch_size=32)

# Step 6: Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print(f'Mean Absolute Error on Test Data: {mae}')

# Step 7: Make predictions on the test data
y_pred = model.predict(x_test)

# Step 8: Calculate and print Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# pratical no.2 : Binary classification using Deep Neural Networks Example: Classify movie reviews into positive&quot; reviews and &quot;negative&quot; reviews.
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Flatten
import matplotlib.pyplot as plt

# Load the IMDB dataset (top 10,000 most frequent words)
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Set maximum sequence length for padding
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Build the deep neural network model
model = Sequential([
    Embedding(vocab_size, 32, input_length=max_length),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # LSTM layer for sequential data processing
    Dense(32, activation='relu'),
    Dropout(0.5),  # Dropout for regularization
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(x_train, y_train, epochs=4, batch_size=64, validation_split=0.2, verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()



# practical no.3 : Convolutional neural network (CNN) (Any One from the following) Use any dataset of plant disease and design a plant disease detection system using CNN. Use MNIST Fashion Dataset and create a classifier to classify fashion

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train.shape

x_test.shape

cat = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']
set(y_train)

plt.imshow(x_train[425], cmap = 'gray');
plt.figure(figsize=(16,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(cat[y_train[i]])

x_train[1];
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train[1].shape

# add a colour channel
x_train = np.expand_dims(x_train, axis = -1)
x_test = np.expand_dims(x_test, axis = -1)
x_train[1].shape

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#define the model architecture
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
model = Sequential([
    Conv2D(32, (3,3), activation= 'relu', input_shape=(28, 28, 1)),
    MaxPool2D((2,2)),

    Conv2D(64, (3,3), activation= 'relu'),
    MaxPool2D((2,2)),

    Conv2D(64, (3,3), activation= 'relu'),

    Flatten(),

    Dense(64, activation= 'relu'),
    Dense(10, activation= 'softmax')
])
model.summary()

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'],
             optimizer = 'adam')
history = model.fit(x_train, y_train, epochs = 10, batch_size = 10, 
                  validation_split= 0.2)
loss, accuracy = model.evaluate(x_test, y_test)

# practical no. 4 : Basic Image Processing - loading images, Cropping, Resizing, Thresholding, Contour analysis, Bolb detection

import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # Colab-compatible image display

# Load image
img = cv2.imread("img/img.png")
cv2_imshow(img)  # Display original image

# Resizing image
img_resize = cv2.resize(img, (300, 300))
cv2_imshow(img_resize)

# Cropping Image
img_crop = img[100:300, 200:500]
cv2_imshow(img_crop)

# Apply binary threshold
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresholded_img = cv2.threshold(gray_img, 125, 255, cv2.THRESH_BINARY)
cv2_imshow(thresholded_img)

# Contour analysis
contours, _ = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
cv2_imshow(contour_img)

# Blob detection (SimpleBlobDetector)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100  # Adjust as needed

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(thresholded_img)

# Draw detected blobs as red circles
blob_img = img.copy()
cv2.drawKeypoints(blob_img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2_imshow(blob_img)

# Release resources
cv2.destroyAllWindows()


# practical no. 5 : Image Annotation – Drawing lines, text circle, rectangle, ellipse on images

import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # Colab-compatible image display

# Load image
img = cv2.imread('img.png')

# Drawing functions
def draw_line(img, pt1, pt2, color, thickness):
    cv2.line(img, pt1, pt2, color, thickness)

def draw_text(img, text, pt, color, thickness):
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

def draw_circle(img, center, radius, color, thickness):
    cv2.circle(img, center, radius, color, thickness)

def draw_rectangle(img, pt1, pt2, color, thickness):
    cv2.rectangle(img, pt1, pt2, color, thickness)

def draw_ellipse(img, center, axes, angle, color, thickness):
    cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness)

# Annotate image
draw_line(img, (100, 100), (200, 200), (255, 0, 0), 2)  # Blue line
draw_text(img, 'Hello', (50, 50), (0, 255, 0), 2)        # Green text
draw_circle(img, (300, 300), 50, (0, 0, 255), 2)         # Red circle
draw_rectangle(img, (400, 100), (500, 200), (255, 255, 0), 2)  # Yellow rectangle
draw_ellipse(img, (200, 400), (50, 100), 45, (0, 255, 255), 2)  # Cyan ellipse

# Display annotated image
cv2_imshow(img)

# Save annotated image
cv2.imwrite('annotated_image.jpg', img)

# Release resources
cv2.destroyAllWindows()


# practical no. 6 : create a basic game where the player can move the characterr using arrow keys or WASD use unity

using UnityEngine;

public class Scripting : MonoBehaviour
{
    public float xspeed = 0.1f;
    public float yspeed = 0.05f; // Added missing semicolon here

    void Start()
    {
        // Initialization if needed
    }

    void Update()
    {
        if (Input.GetKey(KeyCode.D))
        {
            transform.position += new Vector3(xspeed, 0, 0);
        }
        if (Input.GetKey(KeyCode.A))
        {
            transform.position -= new Vector3(xspeed, 0, 0);
        }
        if (Input.GetKey(KeyCode.S))
        {
            transform.position -= new Vector3(0, yspeed, 0);
        }
        if (Input.GetKey(KeyCode.W))
        {
            transform.position += new Vector3(0, yspeed, 0);
        }
    }
}
