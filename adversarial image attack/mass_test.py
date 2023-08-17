# USAGE
# python mass_test.py --input pig.jpg --output adversarial --delta delta --class-idx 341 --target-idx 342

# import necessary packages
from pyimagesearch.utils import get_class_idx

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.applications import VGG16
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


def generate_adversaries(model, baseImage, delta, classIdx, targetIdx, steps=50):
	# iterate over the number of steps
	for step in range(0, steps):
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
			
			loss =  -0.2*-sccLoss(tf.convert_to_tensor([classIdx]),
				predictions) + 0.8*sccLoss(tf.convert_to_tensor([targetIdx]),
				predictions)
#
			# check to see if we are logging the loss value, and if
			# so, display it to our terminal
			if step % 5 == 0:
				print("step: {}, loss: {}...".format(step,
					loss.numpy()))

		# calculate the gradients of loss with respect to the
		# perturbation vector
		gradients = tape.gradient(loss, delta)

		# update the weights, clip the perturbation vector, and
		# update its value
		optimizer.apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=EPS))

	# return the perturbation vector
	return delta
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
ap.add_argument("-t", "--target-idx", type=int, required=True,
	help="ImageNet class ID of the targeted label")
args = vars(ap.parse_args())

# define the epsilon and learning rate constants
EPS = 2 / 255.0
LR = 0.1

# load the input image from disk and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args["input"])
image = preprocess_image(image)

# load the pre-trained ResNet50 model for running inference
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")
model2 = MobileNetV2(include_top=True,weights='imagenet')

# initialize optimizer and loss function


# create a tensor based off of the input image and initialize the
# perturbation vector (we will update this vector via training)


f = open('result282.csv', 'w')
writer = csv.writer(f)


# generate the perturbation vector to create an adversarial example
# j is the range of image prediciton labels we want to test on
for j in range(0, 10):
	print(j)
	optimizer = Adam(learning_rate=LR)
	sccLoss = SparseCategoricalCrossentropy()
	baseImage = tf.constant(image, dtype=tf.float32)
	delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)
	print("[INFO] generating perturbation...")
	deltaUpdated = generate_adversaries(model, baseImage, delta,
		args["class_idx"], j)

	# create the adversarial example, swap color channels, and save the
	# output image to disk
	print("[INFO] creating adversarial example...")
	adverImage = (baseImage + deltaUpdated).numpy().squeeze()
	adverImage = np.clip(adverImage, 0, 255).astype("uint8")
	adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
	deltaImage = (deltaUpdated * 100).numpy().squeeze()
	deltaImage = np.clip(deltaImage, 0, 255).astype("uint8")
	deltaImage = cv2.cvtColor(deltaImage, cv2.COLOR_RGB2BGR)
	cv2.imwrite(args["output"]+str(j)+".png", adverImage)
	cv2.imwrite(args["delta"]+str(j)+".png", deltaImage)

	# run inference with this adversarial example, parse the results,
	# and display the top-1 predicted result
	print("[INFO] running inference on the adversarial example...")
	preprocessedImage = preprocess_input(baseImage + deltaUpdated)
	predictions = model.predict(preprocessedImage)
	predictions = decode_predictions(predictions)[0]
	temp = [-1, -1,0,-1,-1,0]
	for (i, (imagenetID, label, prob)) in enumerate(predictions):
		if (get_class_idx(label) == j):
			temp[0:3] = [i, label, prob]
	predictions2 = model2.predict(preprocessedImage)
	predictions2 = decode_predictions(predictions2)[0]
	print(predictions2)
	
	for (i, (imagenetID, label, prob)) in enumerate(predictions2):
		
		if (get_class_idx(label) == j):
			temp[3:] = [i, label, prob]
	writer.writerow(temp)
f.close()

    
		
