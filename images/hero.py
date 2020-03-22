from pickle import load

from numpy import argmax

from keras.preprocessing.sequence import pad_sequences

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.models import Model

from keras.models import load_model

import cv2 as cv 

from matplotlib.image import imread

import re

from imutils import build_montages

from imutils import paths

import numpy as np

import random

import matplotlib.pyplot as plt





# extract features from each photo in the directory

def extract_features(filename):

	# load the model

	model1 = VGG16()

    model = VGG16()

    

    model1.layers.pop()

    

    model1= Model(inputs = model1.inputs, outputs = model1.layers[-1].output)

    

    image = load_img(file,target_size=(224,224))

    

    image = img_to_array(image)

    

    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))

    

    image = preprocess_input(image)

    

    feature = model1.predict(image)

    

    yhat = model.predict(image)

    

    label = decode_predictions(yhat)

# retrieve the most likely result, e.g. highest probability

    label = label[0][0]

# print the classification

    print('%s (%.2f%%)' % (label[1], label[2]*100))

    return feature, label



# map an integer to a word

def word_for_id(integer, tokenizer):

	for word, index in tokenizer.word_index.items():

		if index == integer:

			return word

	return None



# generate a description for an image

def generate_desc(model, tokenizer, photo, max_length):

	# seed the generation process

	in_text = 'startseq'

	# iterate over the whole length of the sequence

	for i in range(max_length):

		# integer encode input sequence

		sequence = tokenizer.texts_to_sequences([in_text])[0]

		# pad input

		sequence = pad_sequences([sequence], maxlen=max_length)

		# predict next word

		yhat = model.predict([photo,sequence], verbose=0)

		# convert probability to integer

		yhat = argmax(yhat)

		# map integer to word

		word = word_for_id(yhat, tokenizer)

		# stop if we cannot map the word

		if word is None:

			break

		# append as input for generating the next word

		in_text += ' ' + word

		# stop if we predict the end of the sequence

		if word == 'endseq':

			break

	return in_text



tokenizer = load(open('tokenizer.pkl','rb'))



max_length = 34



model = load_model('model_11.h5')



imagePaths = list(paths.list_images('images/'))

random.shuffle(imagePaths)

imagePaths = imagePaths[:20]



results = []



for p in imagePaths:

    photo, label = extract_features(p)

    description = generate_desc(model, tokenizer, photo, max_length)

    orig = cv.imread(p)

    color = (0, 0, 255)

    orig = cv.resize(orig, (700, 700))

    cv.putText(orig, label[1], (3, 50), cv.FONT_HERSHEY_SIMPLEX, 1,color, 2)

    cv.putText(orig, description, (3, 680), cv.FONT_HERSHEY_SIMPLEX, 1,color, 2)

    results.append(orig)

montage = build_montages(results, (450, 450), (3, 1))[0]



# show the output montage

cv.imshow("Results", montage)

cv.waitKey(0)


