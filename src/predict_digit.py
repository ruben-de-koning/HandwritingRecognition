import tensorflow as tf
import numpy as np
import cv2

#Load the trained model
model = tf.keras.models.load_model('model/mnist_digit_recognition.h5')

#Load an image of a digit
image_path = 'data/digits/digit4.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#Preprocess the image
img = cv2.resize(img, (28,28))
img = np.expand_dims(img, axis=0)
img = img / 255.0

#Make the prediction
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

print(f"The model predicted: {predicted_digit}")