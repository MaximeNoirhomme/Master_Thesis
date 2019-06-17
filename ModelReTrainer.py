import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense
import dataHandler as bh
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
import myUtils as mu
from visual_backprop import VisualBackprop
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
import matplotlib.cm as cm
from keras import activations
from numpy.random import seed as s
from sklearn.utils import class_weight
import os
from test_vis import VisualizeImageMaximizeFmap
from keras.applications import resnet50
from keras import backend as K
import pandas as pd
# Constants definition:

# Only retrain the last layer.
LAST_LAYER_TRAIN_MODE = 0
# retrain the last layer plus all the weight of the network.
FULL_TRAIN_MODE = 1

# Class that is responsible to retrain a model in two different possible modes:
# 
#	- Only the last layer of the network (LAST_LAYER_TRAIN_MODE).
#	- The all network (FULL_TRAIN_MODE).

class ModelReTrainer:
	def __init__(self, archi_model, trainer_mode, dataHandler, nb_classes = 25, path_weight = None, path_save = None, already_weighted = False, seed=7, 
				optimizer = Adam(), path_logger = None, trainoff = False): #SGD(lr = 0.001, momentum = 0.9)
		s(seed)
		self.optimizer = optimizer
		# Either set to LAST_LAYER_TRAIN_MODE or FULL_TRAIN_MODE.
		self.trainer_mode = trainer_mode
		# Path that indicates where to find the Weight. If None, train from scratch.
		self.path_weight = path_weight
		self.nb_classes = nb_classes
		# Whether the weights have already been introduced in the model architecture. 
		self.already_weighted = already_weighted
		# DataHandler is a class that is responsible for the accesing of data.
		self.dataHandler = dataHandler
		# Path where to save the model during checkpoints, if set to None, there is no checkpoint. 
		self.path_save = path_save
		self.trainoff = trainoff
		# Model to retrain (architecture + weight)
		self.model = self._load_model(archi_model)
		self.visual_backprop = None
		self.path_logger = path_logger

	def get_loss(self, weights):	

		def weighted_loss(y_true, y_pred):# https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
			# scale predictions so that the class probas of each sample sum to 1
			y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
			# clip to prevent NaN's and Inf's
			y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
			# calc
			loss = y_true * K.log(y_pred) * weights
			loss = -K.sum(loss, -1)
			return loss
    
		return weighted_loss		
	

	def _load_model(self, archi_model):
		'''
			Load the weights of the models (if not exists, initialize them) then
			associate it with the architecture archi_model.
		
			Parameters:
				- archi_model: it is the architecture (not compiled) of the model.
			
			Return:
				- The model to be retrained.
		'''	

		model = archi_model

		if self.path_weight == None and not self.already_weighted:
			# TODO 
			pass
		
		else:
			# Removing the last layer
			model.layers.pop()

			# Retrieve the penultimate layer of the original network. 
			out = model.layers[-1].output
			# Give it as input to the "new last layer" corresponding to new labels.
			out = Dense(self.nb_classes, activation="softmax")(out)

			model = Model(inputs = model.input, outputs=out)

			# load the weights if not already loaded.
			if not self.already_weighted:
				print(self.path_weight)
				model.load_weights(self.path_weight)

		if self.trainer_mode == LAST_LAYER_TRAIN_MODE:
			for layer in model.layers[:-2]:
				layer.trainable = False

				model.layers[-1].trainable = True
				model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
		else:
			for layer in model.layers:
				layer.trainable = True

			if self.trainoff:
				weights = np.zeros((self.nb_classes, 1)) 
			else:
				weights = self.dataHandler.get_balanced_weights(0)

			model.compile(optimizer=self.optimizer, loss=self.get_loss(weights), metrics=['accuracy'])

		return model
		
	def train(self, start_epoch):
		'''
			(Re)Trains the model of the class for a given number of epochs.
			Parameters:
				- epochs: Integer that corresponds to number of epochs to train the model.
				(The definition of an epoch is the same than keras documentation). #TODO: remake the specification

		'''

		early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')	
		callbacks = [early_callback]

		if self.path_save != None:
			cp_callback = tf.keras.callbacks.ModelCheckpoint(self.path_save, 
                                                 save_weights_only=True,
                                                 verbose=1,
												 save_best_only=True)

			callbacks.append(cp_callback)

		if self.path_logger != None:
			csv_logger = tf.keras.callbacks.CSVLogger(self.path_logger)
			callbacks.append(csv_logger)			
		
		history = self.model.fit_generator(self.dataHandler.generator(bh.TRAIN_TYPE), steps_per_epoch=self.dataHandler.get_size(bh.TRAIN_TYPE) // self.dataHandler.batch_size,
			epochs=25, verbose=1, callbacks=callbacks, validation_data=self.dataHandler.generator(bh.VALIDATION_TYPE),
			validation_steps=self.dataHandler.get_size(bh.VALIDATION_TYPE) // self.dataHandler.batch_size, class_weight=None, max_queue_size=10, workers=1,
			 use_multiprocessing=False, shuffle=True, initial_epoch=start_epoch, )
		
		return history
		  
	def evaluate(self):
		return self.model.evaluate_generator(self.dataHandler.generator(bh.TESTING_TYPE), steps=self.dataHandler.get_size(bh.TESTING_TYPE) // 1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

	def predict(self, img_path):
		i = np.expand_dims(img_to_array(load_img(img_path, target_size=(224, 224))), axis=0)
		processed_img = resnet50.preprocess_input(i.copy())
		processed_img *= 1/255
		
		return self.model.predict(processed_img)
		
	def save(self, path_weight):
		self.model.save_weights(path_weight)
	
	def confusion_matrix(self, data_handler=None, path_matrix=None):
		
		data_handler = data_handler if data_handler != None else self.dataHandler
		for test_csv_path in data_handler.test_csv_paths:
			test_datagen = ImageDataGenerator(rescale=1./255)

			print(test_csv_path)
			df=pd.read_csv(test_csv_path)
			gen = test_datagen.flow_from_dataframe(dataframe=df,
						directory=None,
						x_col="img_path",
						y_col="label",
						class_mode="categorical",
						target_size=(224, 224),
						batch_size=1,
						shuffle=False)
			
			Y_pred = self.model.predict_generator(gen, mu.get_size_csv(test_csv_path) - 1)            

			y_pred = np.argmax(Y_pred, axis=1)
			
			score_macro = f1_score(gen.classes, y_pred, average='macro')
			score_micro = f1_score(gen.classes, y_pred, average='micro')
			print((score_macro, score_micro))
			
			conf_matrix = confusion_matrix(gen.classes, y_pred)
			print(conf_matrix)	
			if path_matrix != None:
				np.save(path_matrix, conf_matrix)

			return conf_matrix

	def visualise_network(self, img_path, arch_type='resnet50', output='vis', save=True, im=None):
		model = self.model

		self.visual_backprop = VisualBackprop(model, arch_type=arch_type)
		
		if im != None:
			im = load_img(img_path, target_size=(224,224))
			
		img = np.asarray(im)
		x = np.expand_dims(img, axis=0)	
		
		mask = self.visual_backprop.get_mask(x[0])
		
		if save:
			filename, _ = os.path.splitext(output)
			mu.show_image(mask, ax=plt.subplot('111'), title='ImageNet VisualBackProp', output=output)
			im.save(filename+'_original'+'.jpg')

		return mask