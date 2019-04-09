import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense
import dataHandler as bh
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from keras.preprocessing.image import ImageDataGenerator
import myUtils as mu
from visual_backprop import VisualBackprop
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
#from vis.visualization import visualize_cam
import matplotlib.cm as cm
# https://github.com/raghakot/keras-vis/blob/master/examples/resnet/attention.ipynb
#from vis.utils import utils
from keras import activations
#from vis.visualization import visualize_saliency, overlay
from numpy.random import seed as s
from sklearn.utils import class_weight
import os
from test_vis import VisualizeImageMaximizeFmap
from keras.applications import resnet50
from keras import backend as K
import pandas as pd
#ImageFile.LOAD_TRUNCATED_IMAGES = True
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
				optimizer = SGD(lr = 0.001, momentum = 0.9), path_logger = None):
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
		# Model to retrain (architecture + weight)
		self.model = self._load_model(archi_model)
		self.visual_backprop = None
		self.path_logger = path_logger

	def get_loss(self, weights):	
		print(weights)
		print(sum(weights))
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
	
	"""def get_metrix(self, weights):
		weights = tf.constant(weights, dtype=tf.float32)
		def c_w_acc(y_true, y_pred):
			true_label_index = K.argmax(y_true, axis=-1)
			weight = tf.gather(weights, true_label_index)

			'''return weight*K.cast(K.equal(true_label_index,
							K.argmax(y_pred, axis=-1)), K.floatx())'''
			print(y_true)
			print(y_pred)
			return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  		K.argmax(y_pred, axis=-1)))

		return c_w_acc"""
	
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
			print(model.input)
			print(model.layers[-1].output)
			model = Model(inputs = model.input, outputs=out)

			# load the weights if not already loaded.
			if not self.already_weighted:
				model.load_weights(self.path_weight)
			
		# TODO: changer les param de compile
		if self.trainer_mode == LAST_LAYER_TRAIN_MODE:
			for layer in model.layers[:-2]:
				layer.trainable = False

				model.layers[-1].trainable = True
				model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
		else:
			for layer in model.layers:
				layer.trainable = True

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
			epochs=1, verbose=1, callbacks=callbacks, validation_data=self.dataHandler.generator(bh.VALIDATION_TYPE),
			validation_steps=self.dataHandler.get_size(bh.VALIDATION_TYPE) // self.dataHandler.batch_size, class_weight=None, max_queue_size=10, workers=1,
			 use_multiprocessing=False, shuffle=True, initial_epoch=start_epoch, )
		
		return history
		  
	def evaluate(self):
		return self.model.evaluate_generator(self.dataHandler.generator(bh.TESTING_TYPE), steps=self.dataHandler.get_size(bh.TESTING_TYPE) // 1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

	'''def evaluate_w_test_dataset(self, dataHandler):
		return self.model.evaluate_generator(dataHandler.generator(bh.TESTING_TYPE), steps=dataHandler.get_size(bh.TESTING_TYPE) // 1 , max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
	'''

	def predict(self, img_path):
		i = np.expand_dims(img_to_array(load_img(img_path, target_size=(224, 224))), axis=0)
		processed_img = resnet50.preprocess_input(i.copy())
		processed_img *= 1/255
		
		return self.model.predict(processed_img)
		
	def save(self, path_weight):#, path_model):
		#self.model.save(path_model)
		self.model.save_weights(path_weight)
	
	def confusion_matrix(self, data_handler=None, path_matrix=None):
		'''data_handler = data_handler if data_handler != None else self.dataHandler
		for path, path_test in zip(data_handler.paths, data_handler.test_paths):
			test_datagen = ImageDataGenerator(rescale=1./255)

			test_data_path = path + '/' + path_test
			print(test_data_path)
			gen = test_datagen.flow_from_directory(test_data_path,
													target_size=(224, 224),
													batch_size=1,
													class_mode='categorical',
													shuffle=False)#, classes=data_handler.get_labels(1))
			print(gen.class_indices)
			Y_pred = self.model.predict_generator(gen, data_handler.get_size(bh.TESTING_TYPE))
			y_pred = np.argmax(Y_pred, axis=1)

			score_macro = f1_score(gen.classes, y_pred, average='macro')
			score_micro = f1_score(gen.classes, y_pred, average='micro')
			print((score_macro, score_micro))
			
			conf_matrix = confusion_matrix(gen.classes, y_pred)
			print(conf_matrix)	
			if path_matrix != None:
				np.save(path_matrix, conf_matrix)

			return conf_matrix
		'''
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
			
			Y_pred = self.model.predict_generator(gen, data_handler.get_size(bh.TESTING_TYPE))            
			#Y_pred = Y_pred[0]#[1:]
			Y_pred = Y_pred[1:]
			print(Y_pred.shape)
			#print("ok3 ", Y_pred.shape)
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

	# TODO: debug kernel_vis + gig or delete them.
	def kernel_vis(self):
		max_nfmap = np.Inf ## print ALL the images
		self.model.summary()
		input_img = self.model.layers[0].input
		layer_names = ['conv1']#["conv2d_22", "conv2d_23", "conv2d_24"]
		layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
		visualizer = VisualizeImageMaximizeFmap((224,224,1), self.model)
		print("find images that maximize feature maps")
		argimage = visualizer.find_images(input_img,
										layer_names,
										layer_dict, 
										max_nfmap)
		print("plot them...")
		visualizer.plot_images_wrapper(argimage,n_row = 8, scale = 1)

	def gif(self, label):
		from vis.losses import ActivationMaximization
		from vis.regularizers import TotalVariation, LPNorm
		from vis.input_modifiers import Jitter
		from vis.optimizer import Optimizer
		from keras.applications.vgg16 import VGG16
		from vis.callbacks import GifGenerator
		# Build the VGG16 network with ImageNet weights
		model = VGG16(weights='imagenet', include_top=True)#self.model
		print('Model loaded.')

		# The name of the layer we want to visualize
		# (see model definition in vggnet.py)
		layer_name = 'predictions'#'dense_1'
		layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
		print(layer_dict)
		output_class = [label]

		losses = [
			(ActivationMaximization(layer_dict[layer_name], output_class), 2),
			(LPNorm(model.input), 10),
			(TotalVariation(model.input), 10)
		]
		opt = Optimizer(model.input, losses)
		opt.minimize(max_iter=500, verbose=True, image_modifiers=[Jitter()], callbacks=[GifGenerator('opt_progress')])

'''from dataHandler import *
from keras.applications import resnet50
model = ResNet50(weights='imagenet')
dh = DataHandler(['dataset_csv_path/0-s7', 'dataset_csv_path/1-s7p0.1'], ['dataset_csv_path/7-s7'])
ModelReTrainer(model, FULL_TRAIN_MODE, dh, path_weight='weights/0_0_1-ws7p0.1').confusion_matrix(path_matrix='conf_matrix/0_0_1&7-ws7.npy')#.train(start_epoch=0)'''