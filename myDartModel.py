from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Conv2D, MaxPool2D, Input, GlobalAveragePooling2D, Reshape
from keras.models import Model, Sequential
from dartDataHandler import *
import tensorflow as tf
from keras.layers import Input, Lambda
from tensorflow import keras
from myLayers import *
from flipGradientTF import *
from keras.utils import plot_model
from myCallbacks import *
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import dataHandler as bh
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from keras.optimizers import SGD
from keras.initializers import TruncatedNormal, Constant

from keras.preprocessing.image import load_img, img_to_array
import os

class MyDartModel:
    def __init__(self, source_prop, nb_classes, dartDataHandler, nb_epoch, l0, gamma, alpha, testing=False, with_kronecker=True, complex_m=False):
        '''
            Constructor of MyDartModel
            Parameters:
                - nb_classes: Integer, the total number of classes in the classification.
                - dartDataHandler: DartDataHandler, DartDataHandler that provides the generator used for training/testing
                - nb_epoch: Integer, maximum number of epoch.
                - l0: Double, l0 value (see report)
                - gamma: Double, gamma value (see report)
                - with_kronecker: Bool, if set to true, kronecker layer is used.

            Notice that the constant lambda of the reversal gradient is equalled to l0 * (2/(1  - exp(-gamma * q)) - 1 ) where q = i * (1 / nb_epoch)
            where i is the number of the current epoch.
        '''
        
        self.nb_classes = nb_classes
        self.dartDataHandler = dartDataHandler
        # Input and output of the feature extractor
        self._input_extractor = None
        self._output_extractor = None
        # Output of the label classifier
        self._label_classifier = None
        # Output of the domain classifier
        self._domain_classifier = None

        self.nb_epoch = nb_epoch
        self.l0 = l0
        self.gamma = gamma
        self.with_kronecker = with_kronecker
        self.alpha = alpha

        self.complex_m = complex_m
        self.source_prop = source_prop
        # Get model.
        self.model = self.get_model(testing=testing)

    def weighted_loss(self, y_true, y_pred, weights):# https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
        '''
            Define weighted cross-entropy
        '''
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    def get_loss_label(self, target_domain, weights, alpha):
        '''
            Define loss of the label classifier. If the target_domain is source, then
            the loss is the weighted cross-entropy, otherwise it is weighted entropy.
        '''
        def loss_label_fct(target_label, output_label): 
            # if the domain is source then apply cross-entropy, else apply entropy loss function.
            cond = tf.equal(target_domain, tf.fill(tf.shape(target_domain), 0.0))

            true_f = tf.reshape(self.weighted_loss(target_label, output_label, weights), [-1, 1])
            true_f = K.in_train_phase(self.source_prop*true_f, true_f)

            false_f = tf.reshape(self.weighted_loss(output_label, output_label, weights), [-1, 1])#alpha*
            false_f = K.in_train_phase(0*false_f, false_f)

            loss = tf.where(cond, true_f, false_f)
            '''if alpha != 0:
                loss = tf.where(cond, true_f, false_f)
            else:
                loss = tf.where(cond, true_f, true_f)'''
                
            return loss

        return loss_label_fct

    def get_loss_domain(self):
        '''
            Define loss of the domain classifier. It is the weighted binary cross_entropy.
        '''
        def loss_domain_fct(true_domain, pred_domain):
            return keras.losses.categorical_crossentropy(true_domain, pred_domain)

        return loss_domain_fct

    def get_model(self, testing=False):        
        '''
            Define the dart model.
            Parameters:
                - testing: bool, if set to False, the model takes as inputs, the images, the label and the domain (source or target).
                Otherwise, it only takes the images as input.

            Notice that the kronecker layer is there only if self.kronecker is set to True.
        '''
        
        label_input = Input(shape=(self.nb_classes, ), name='label')
        domain_input = Input(shape=(1, ), name='domain')

        input_extractor, output_extractor = self.feature_extractor()

        label_classifier = self.label_classifier_submodel(output_extractor, testing)
        if testing:
            bml = label_classifier
        else:
            bml = BinMultiplexerLayer()([label_classifier, label_input, domain_input])
        
        if self.with_kronecker:
            kpl = KroneckerProductLayer()([output_extractor, bml])
        else:
            kpl = output_extractor
        
        fr = GradientReversal(self.l0, self.gamma, name='gradient_reversal_1')(kpl)

        domain_classifier = self.domain_classifier_submodel(fr, testing)
        if testing:
            self.model = Model([input_extractor], [label_classifier, domain_classifier]) 
        else:
            self.model = Model([input_extractor, label_input, domain_input], [label_classifier, domain_classifier])

        weights = self.dartDataHandler.get_balanced_weights(0)
        
        losses = {
            'lab_class':self.get_loss_label(domain_input, weights, self.alpha),
            'dom_class':self.get_loss_domain(),
        }

        metric = {
            'lab_class': ['accuracy'],
            'dom_class': 'accuracy'
        }

        self.model.compile(optimizer= SGD(lr = 0.001, momentum = 0.9), loss=losses, metrics=metric) #, self.get_metrix(weights) 
        
        '''
        if testing:
            plot_model(self.model, to_file='test_model_testing5.png')
        else:
            plot_model(self.model, to_file='sans_Dart_2.png')'''

        return self.model

    def feature_extractor(self):
        '''
            Define the feature_extractor.
            Return:
                - The feature extractor
        '''
        if self._output_extractor == None:
            '''self._input_extractor = Input(shape=(28, 28, 3))

            #x = Lambda(lambda x: (x - pixel_mean) / 255.)(self._input_extractor)
            x = Conv2D(32, 5, activation='relu', padding='same', kernel_initializer=TruncatedNormal(stddev=0.1), bias_initializer=Constant(0.1))(self._input_extractor)
            x = MaxPool2D(strides=(2,2))(x)
            x = Conv2D(48, 5, activation='relu', padding='same', kernel_initializer=TruncatedNormal(stddev=0.1), bias_initializer=Constant(0.1))(x)
            x = MaxPool2D(strides=(2,2))(x)
            self._output_extractor = Reshape((7*7*48,))(x) #GlobalAveragePooling2D()(x)'''

            # Get resnet50 architecture without the last softmax layer
            resnet = ResNet50(weights='imagenet')
            resnet.layers.pop()
            
            # Set output and input of the feature extractor
            self._output_extractor = resnet.layers[-1].output
            self._input_extractor = resnet.layers[0].input
            # Rename the output layer
            resnet.layers[-1].name = 'output_extractor' 
               
        return self._input_extractor, self._output_extractor

    def label_classifier_submodel(self, input_layer, testing):
        '''
            Define the label classifier.
            Parameters:
                - input_layer: Tensor, the input of the label classifier.
        '''
        if self._label_classifier == None:
            '''x = Dense(100, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.1), bias_initializer=Constant(0.1))(input_layer)
            x = Dense(100, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.1), bias_initializer=Constant(0.1))(x)

            if testing:
                self._label_classifier = Dense(10, activation="softmax", name='lab_class')(x)
            else:
                self._label_classifier = Dense(10, kernel_initializer=TruncatedNormal(stddev=0.1), bias_initializer=Constant(0.1), name='lab_class')(x)'''
            
            self._label_classifier = Dense(self.nb_classes, activation="softmax", name='lab_class')(input_layer)

        return self._label_classifier

    def domain_classifier_submodel(self, input_layer, testing):
        '''
            Define the domain classifier.
            Parameters:
                - input_layer: Tensor, the input of the domain classifier.
        '''   
        if self._domain_classifier == None:
            '''x = Dense(100, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.1), bias_initializer=Constant(0.1))(input_layer)
            #x = GlobalAveragePooling2D()(x)
            #x = Dense(100, activation='relu')(x)
            if testing:
                self._domain_classifier = Dense(2, activation='softmax', name='dom_class')(x)
            else:
                self._domain_classifier = Dense(2, name='dom_class', kernel_initializer=TruncatedNormal(stddev=0.1), bias_initializer=Constant(0.1))(x)'''

            x = Dense(100, activation='relu')(input_layer)

            if self.complex_m:
                x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)

            self._domain_classifier = Dense(2, activation='softmax', name='dom_class')(x)

        return self._domain_classifier

    def train(self, start_epoch = 0, nb_epoch = 25, early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_lab_class_loss', patience=7, verbose=1, mode='auto'), 
                checkpoint_path = None, path_logger = None, mu0 = 0.001, alpha = 10, beta = 0.75):
        '''
            Train the model defined in self.get_model.
            Parameters:
                - start_epoch: Integer, the number of the first epoch (usefull when a training has been stopped before ending)
                - nb_epoch: Integer, maximum number of epoch
                - early_stopping: Keras callbacks, the callback that handle the early stopping (if set to None, there is no early stopping)
                - checkpoint_path: String, path where the checkpoints are saved (if set to None, no checkpoint is made)
                - path_logger: String, path where to save the csv logger (if set to None, no checkpoint is made)
            Return:
                The history of the training.
        '''
        
        step_per_epoch = self.dartDataHandler.get_size(TRAIN_TYPE) // self.dartDataHandler.batch_size
        callbacks = [dartParamUpdate(nb_epoch, step_per_epoch, mu0, alpha, beta)]
        if early_stopping != None:
            callbacks.append(early_stopping)

        if checkpoint_path != None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                        monitor='val_lab_class_loss',#
                                        save_weights_only=True,
                                        verbose=1,
                                        save_best_only=True)
            
            callbacks.append(cp_callback)

        if path_logger != None:    
            csv_logger = tf.keras.callbacks.CSVLogger(path_logger)
            callbacks.append(csv_logger)

        history = self.model.fit_generator(self.dartDataHandler.generator(TRAIN_TYPE), steps_per_epoch=step_per_epoch,
			epochs=nb_epoch, verbose=2, callbacks=callbacks, validation_data=self.dartDataHandler.generator(VALIDATION_TYPE),
			validation_steps=self.dartDataHandler.get_size(VALIDATION_TYPE) // self.dartDataHandler.batch_size, class_weight=None, max_queue_size=10, workers=1,
			 use_multiprocessing=False, shuffle=True, initial_epoch=start_epoch)
       
        return history

    def confusion_matrix(self, data_handler=None, path_matrix_label=None, path_matrix_domain=None, testing=True):
        '''
            Compute the confusion matrixes for both label and domain classifier.
            Parameters:
                - data_handler: DartDataHandler, datahandler used for having testing data (if set to None, the dataHandler of this classe
                is used instead).
                - path_matrix: String, path to save the confusion matrix (if set to None, the confusion matrix is not saved)

            Return:
                - A tuple of numpy array of 2 dimensions, confusion matrixes for both label and domain classifier.
            
        '''
        data_handler = data_handler if data_handler != None else self.dartDataHandler
        label_conf_matrixes = []
        domain_conf_matrixes = []
        csv_paths = data_handler.test_csv_paths if testing else data_handler.valid_csv_paths
        for test_csv_path, is_source in zip(csv_paths, data_handler.is_sources):
            test_datagen = ImageDataGenerator(rescale=1./255)

            df=pd.read_csv(test_csv_path)
            gen = test_datagen.flow_from_dataframe(dataframe=df,
                        directory=None,
                        x_col="img_path",
                        y_col="label",
                        class_mode="categorical",
                        target_size=(self.dartDataHandler.width, self.dartDataHandler.height),
                        batch_size=1,
                        shuffle=False,
                        color_mode=self.dartDataHandler.color_mode)
                        
            Y_pred = self.model.predict_generator(gen, mu.get_size_csv(test_csv_path) - 1)            
            label_pred, dom_pred = Y_pred[0], Y_pred[1]
            label_pred, dom_pred = np.argmax(label_pred, axis=1), np.argmax(dom_pred, axis=1)

            score_macro = f1_score(gen.classes, label_pred, average='macro')
            score_micro = f1_score(gen.classes, label_pred, average='micro')
            print('label_pred = ', (score_macro, score_micro))

            dom_classes = np.zeros((len(gen.classes), 1)) if is_source else np.ones((len(gen.classes), 1))
            score_macro = f1_score(dom_classes, dom_pred, average='macro')
            score_micro = f1_score(dom_classes, dom_pred, average='micro')
            print('domain_pred = ', (score_macro, score_micro))

            label_conf_matrix = confusion_matrix(gen.classes, label_pred)
            domain_conf_matrix = confusion_matrix(dom_classes, dom_pred)

            print(label_conf_matrix)
            print('................')
            print(domain_conf_matrix)

            if path_matrix_label != None:
                np.save(path_matrix_label, label_conf_matrix)

            if path_matrix_domain != None:    
                np.save(path_matrix_domain, domain_conf_matrix)

            label_conf_matrixes.append(label_conf_matrix)
            domain_conf_matrixes.append(domain_conf_matrix)

        return label_conf_matrixes, domain_conf_matrixes

    def save(self, path_weight, path_model):
        '''
            Save the model
            Parameters:
                - path_weight: string, path where to save the weights of the model
                - path_model: string, path where to save the model
        '''
        #self.model.save(path_model)
        self.model.save_weights(path_weight)

    def load_weights(self, path_weights):
        '''
            Load the weights into the model.
            Parameters:
                path_weights: String, path where to find the weights.
        '''
        self.model.load_weights(path_weights)