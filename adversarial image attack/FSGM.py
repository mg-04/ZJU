# USAGE
# python FSGM.py --input pig.jpg --output adversarial.png --delta delta.png --class-idx 341

# import necessary packages
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import argparse
import cv2
import csv

def preprocess_image(image):
	# swap color channels, resize the input image, and add a batch
	# dimension
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)

	# return the preprocessed image
	return image

def clip_eps(tensor, eps):
	# clip the values of the tensor to a given range and return it
	return tf.clip_by_value(tensor, clip_value_min=-eps,
		clip_value_max=eps)

def generate_adversaries(model, baseImage, delta, classIdx):
	# iterate over the number of steps
	
		# record our gradients
	with tf.GradientTape() as tape:
			# explicitly indicate that our perturbation vector should
			# be tracked for gradient updates
			tape.watch(delta)

			# add our perturbation vector to the base image and
			# preprocess the resulting image
			adversary = preprocess_input(baseImage + delta)

			# run this newly constructed image tensor through our
			# model and calculate the loss with respect to the
			# *original* class index
			predictions = model(adversary, training=False)
			loss = -sccLoss(tf.convert_to_tensor([classIdx]),
				predictions)
			

			# check to see if we are logging the loss value, and if
			# so, display it to our terminal
		

		# calculate the gradients of loss with respect to the
		# perturbation vector
	gradients = tape.gradient(loss, delta)
	signed_grad = tf.sign(gradients)

		# update the weights, clip the perturbation vector, and
		# update its value
	
	# return the perturbation vector
	return signed_grad

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to original input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output adversarial image")
ap.add_argument("-d", "--delta", required=True,
		help="path to delta image")
ap.add_argument("-c", "--class-idx", type=int, required=True,
	help="ImageNet class ID of the predicted label")
args = vars(ap.parse_args())

# define the epsilon and learning rate constants
EPS = 2 / 255.0
LR = 5

# load the input image from disk and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args["input"])
image = preprocess_image(image)

# load the pre-trained ResNet50 model for running inference
print("[INFO] loading pre-trained ResNet50 model...")
model = tf.keras.applications.ResNet50(weights='imagenet')

# initialize optimizer and loss function
optimizer = Adam(learning_rate=LR)
sccLoss = SparseCategoricalCrossentropy()

# create a tensor based off of the input image and initialize the
# perturbation vector (we will update this vector via training)
baseImage = tf.constant(image, dtype=tf.float32)
delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

# generate the perturbation vector to create an adversarial example
print("[INFO] generating perturbation...")
deltaUpdated = generate_adversaries(model, baseImage, delta,
	args["class_idx"])

# create the adversarial example, swap color channels, and save the
# output image to disk

print("[INFO] creating adversarial example...")
adverImage = (baseImage + deltaUpdated).numpy().squeeze()
adverImage = np.clip(adverImage, 0, 255).astype("uint8")
adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
deltaImage = (deltaUpdated * 100).numpy().squeeze()
deltaImage = np.clip(deltaImage, 0, 255).astype("uint8")
deltaImage = cv2.cvtColor(deltaImage, cv2.COLOR_RGB2BGR)
cv2.imwrite(args["output"], adverImage)
cv2.imwrite(args["delta"], deltaImage)
# run inference with this adversarial example, parse the results,
# and display the top-1 predicted result
print("[INFO] running inference on the adversarial example...")
preprocessedImage = preprocess_input(baseImage + deltaUpdated)
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions, top=3)[0]
label = predictions[0][1]
confidence = predictions[0][2] * 100
print("[INFO] label: {} confidence: {:.2f}%".format(label,
	confidence))

# draw the top-most predicted label on the adversarial image along
# with the confidence score
text = "{}: {:.2f}%".format(label, confidence)
cv2.putText(adverImage, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
	(0, 255, 0), 2)

# show the output image
cv2.imshow("Output", adverImage)
cv2.waitKey(0)
