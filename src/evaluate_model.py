import tensorflow as tf

#Load the trained model
model = tf.keras.models.load_model('model/mnist_digit_recognition.h5')

#Load the test data
mnist = tf.keras.datasets.mnist
(_, _), (test_images, test_labels) = mnist.load_data()

#Normalize test images
test_images = test_images / 255.0

#Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy: ', test_acc)
print('Test loss: ', test_loss)