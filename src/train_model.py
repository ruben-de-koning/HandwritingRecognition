import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

#Download MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

#Build and train the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
trained_model = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
model.save('model/mnist_digit_recognition.h5')

#Plot the training results
plt.plot(trained_model.history['accuracy'], label='accuracy')
plt.plot(trained_model.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')
plt.show()