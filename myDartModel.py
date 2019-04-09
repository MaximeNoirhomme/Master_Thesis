from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Conv2D, MaxPool2D, Input, GlobalAveragePooling2D
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

class MyDartModel: # TODO: add les losses pour ne pas avoir d'erreur:"error when checking target"
    def __init__(self, nb_classes, dartDataHandler, nb_epoch, l0, gamma, testing=False, with_kronecker=True):
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

        # Get model.
        self.model = self.get_model(testing=testing)

    """def get_metrix(self, weights):
        '''
            Definition of custum metrix
        '''
        weights = tf.constant(weights, dtype=tf.float32)
        def c_w_acc(y_true, y_pred):
            true_label_index = K.argmax(y_true, axis=-1)
            weight = tf.gather(weights, true_label_index)

            return weight*K.cast(K.equal(true_label_index,
                            K.argmax(y_pred, axis=-1)),
                    K.floatx())

        return c_w_acc
    """

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
      
    def get_loss_label(self, target_domain, weights):
        '''
            Define loss of the label classifier. If the target_domain is source, then
            the loss is the weighted cross-entropy, otherwise it is weighted entropy.
        '''
        # TODO: add the domain weights instead of hardcoded them (and one for training and one for validation).
        def loss_label_fct(target_label, output_label): 
            # if the target domain is source then apply cross-entropy, else apply entropy loss function.
            cond = tf.equal(target_domain, tf.fill(tf.shape(target_domain), 0.0))

            '''true_f = tf.reshape(keras.backend.categorical_crossentropy(target_label, output_label), [-1, 1])
            false_f = tf.reshape(keras.backend.categorical_crossentropy(output_label, output_label), [-1, 1])'''
            '''weight_target = tf.reshape(tf.fill(tf.shape(target_domain), 15451/(1119*2)), [-1, 1], name='ok2')
            weight_source = tf.reshape(tf.fill(tf.shape(target_domain), 15451/(14332*2)), [-1, 1], name='ok2')'''

            
            true_f = tf.reshape(self.weighted_loss(target_label, output_label, weights), [-1, 1])
            false_f = 0.6*tf.reshape(self.weighted_loss(output_label, output_label, weights), [-1, 1])

            loss = tf.where(cond, true_f, false_f)
            #w = tf.where(cond, weight_source, weight_source)

            return loss

        return loss_label_fct 

    def get_loss_domain(self):
        '''
            Define loss of the domain classifier. It is the weighted binary cross_entropy.
        '''
        # TODO: add the domain weights instead of hardcoded them (and one for training and one for validation).
        def loss_domain_fct(true_domain, pred_domain):
            return 1*keras.backend.binary_crossentropy(true_domain, pred_domain) # beta = 1
            cond = tf.equal(true_domain, tf.fill(tf.shape(true_domain), 0.0)) # if it is source

            pred_domain = K.clip(pred_domain, K.epsilon(), 1 - K.epsilon())
            #weight_target = tf.reshape(K.in_train_phase(tf.fill(tf.shape(true_domain), 110/(10*2)), tf.fill(tf.shape(true_domain), 140/(40*2))), [-1, 1], name='ok2')
            #weight_source = tf.reshape(K.in_train_phase(tf.fill(tf.shape(true_domain), 110/(100*2)), tf.fill(tf.shape(true_domain), 140/(100*2))), [-1, 1], name='ok')

            #weight_target = tf.reshape(tf.fill(tf.shape(true_domain), 15451/(1119*2)), [-1, 1], name='ok2')
            #weight_source = tf.reshape(tf.fill(tf.shape(true_domain), 15451/(14332*2)), [-1, 1], name='ok2')

            weight_target = tf.reshape(tf.fill(tf.shape(true_domain), 49796/(1504*2)), [-1, 1], name='ok2')
            weight_source = tf.reshape(tf.fill(tf.shape(true_domain), 49796/(48292*2)), [-1, 1], name='ok2')

            w0 = tf.where(cond, weight_source, weight_target, name='wo')
            w1 = tf.where(cond, weight_target, weight_source, name='w1')

            loss = -(true_domain * K.log(pred_domain) * w0 + w1 * (1 - true_domain) * K.log((1 - pred_domain)))
            return loss

        return loss_domain_fct

    def get_model(self, testing=False):        
        '''
            Define the dart model.
            Parameters:
                - testing: bool, if set to False, the model takes as inputs, the images, the label and the domain (source or target).
                Otherwise, it only takes the images as input.

            Notice that the kronecker layer is there only if self.kronecker is set to True.
        '''
        
        #os.environ["PATH"] += os.pathsep + 'D:\\Users\\Noirh\\Documents\\TFE\\release\\bin\\dot.exe'
        print(self.nb_classes)
        label_input = Input(shape=(self.nb_classes, ), name='label')
        domain_input = Input(shape=(1, ), name='domain')

        input_extractor, output_extractor = self.feature_extractor()

        label_classifier = self.label_classifier_submodel(output_extractor)
        if testing:
            bml = label_classifier
        else:
            bml = BinMultiplexerLayer()([label_classifier, label_input, domain_input])
        
        if self.with_kronecker:
            kpl = KroneckerProductLayer()([output_extractor, bml])
        else:
            kpl = output_extractor
        
        fr = GradientReversal(self.l0, self.gamma)(kpl)

        domain_classifier = self.domain_classifier_submodel(fr)
        if testing:
            self.model = Model([input_extractor], [label_classifier, domain_classifier]) #
        else:
            self.model = Model([input_extractor, label_input, domain_input], [label_classifier, domain_classifier])

        weights = self.dartDataHandler.get_balanced_weights(0)
        
        losses = {
            'lab_class':self.get_loss_label(domain_input, weights),
            'dom_class':self.get_loss_domain()
        }

        self.model.compile(optimizer='adam', loss=losses, metrics=['accuracy']) #, self.get_metrix(weights) 
        '''if testing:
            plot_model(self.model, to_file='test_model_testing4.png')
        else:
            plot_model(self.model, to_file='test_model4.png')'''

        return self.model

    def feature_extractor(self):
        '''
            Define the feature_extractor.
            Return:
                - The feature extractor
        '''
        if self._output_extractor == None:
            self._input_extractor = Input(shape=(28, 28, 1))
            x = Conv2D(32, 5, activation='relu')(self._input_extractor)
            x = MaxPool2D(strides=(2,2))(x)
            x = Conv2D(48, 5, activation='relu')(x)
            self._output_extractor =  MaxPool2D(strides=(2,2))(x)
            '''# Get resnet50 architecture without the last softmax layer
            resnet = ResNet50(weights='imagenet')
            resnet.layers.pop()
            # Set output and input of the feature extractor
            self._output_extractor = resnet.layers[-1].output
            self._input_extractor = resnet.layers[0].input
            # Rename the output layer
            resnet.layers[-1].name = 'output_extractor'
            '''


        return self._input_extractor, self._output_extractor

    def label_classifier_submodel(self, input_layer):
        '''
            Define the label classifier.
            Parameters:
                - input_layer: Tensor, the input of the label classifier.
        '''
        if self._label_classifier == None:
            x = Dense(100, activation='relu')(input_layer)
            x = Dense(100, activation='relu')(x)
            x = GlobalAveragePooling2D()(x)
            self._label_classifier = Dense(self.nb_classes, activation="softmax", name='lab_class')(x)

        return self._label_classifier

    def domain_classifier_submodel(self, input_layer):
        '''
            Define the domain classifier.
            Parameters:
                - input_layer: Tensor, the input of the domain classifier.
        '''   
        if self._domain_classifier == None:
            x = Dense(100, activation='relu')(input_layer)
            x = GlobalAveragePooling2D()(x)
            self._domain_classifier = Dense(1, activation='sigmoid', name='dom_class')(x)

        return self._domain_classifier

    def train(self, start_epoch = 0, nb_epoch = 25, early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_lab_class_loss', patience=7, verbose=1, mode='auto'), 
                checkpoint_path = None, path_logger = None):
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
        
        
        callbacks = [dartParamUpdate(nb_epoch)]
        if early_stopping != None:
            callbacks.append(early_stopping)

        if checkpoint_path != None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                        monitor='val_lab_class_loss',
                                        save_weights_only=True,
                                        verbose=1,
                                        save_best_only=True)
            
            callbacks.append(cp_callback)

        if path_logger != None:    
            csv_logger = tf.keras.callbacks.CSVLogger(path_logger)
            callbacks.append(csv_logger)

        history = self.model.fit_generator(self.dartDataHandler.generator(TRAIN_TYPE), steps_per_epoch=self.dartDataHandler.get_size(TRAIN_TYPE) // self.dartDataHandler.batch_size,
			epochs=nb_epoch, verbose=1, callbacks=callbacks, validation_data=self.dartDataHandler.generator(VALIDATION_TYPE),
			validation_steps=self.dartDataHandler.get_size(VALIDATION_TYPE) // self.dartDataHandler.batch_size, class_weight=None, max_queue_size=10, workers=1,
			 use_multiprocessing=False, shuffle=True, initial_epoch=start_epoch)

        print(history.history)
       
        return history

    def confusion_matrix(self, data_handler=None, path_matrix=None):
        '''
            Compute the confusion matrix.
            Parameters:
                - data_handler: DartDataHandler, datahandler used for having testing data (if set to None, the dataHandler of this classe
                is used instead).
                - path_matrix: String, path to save the confusion matrix (if set to None, the confusion matrix is not saved)
        '''
        data_handler = data_handler if data_handler != None else self.dartDataHandler
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
            Y_pred = Y_pred[0]
            y_pred = np.argmax(Y_pred, axis=1)
            
            score_macro = f1_score(gen.classes, y_pred, average='macro')
            score_micro = f1_score(gen.classes, y_pred, average='micro')
            print((score_macro, score_micro))
            
            conf_matrix = confusion_matrix(gen.classes, y_pred)
            print(conf_matrix)	
            if path_matrix != None:
                np.save(path_matrix, conf_matrix)

        return conf_matrix

    def save(self, path_weight, path_model):
        '''
            Save the model
            Parameters:
                - path_weight: string, path where to save the weights of the model
                - path_model: string, path where to save the model
        '''
        self.model.save(path_model)
        self.model.save_weights(path_weight)

    def load_weights(self, path_weights):
        '''
            Load the weights into the model.
            Parameters:
                path_weights: String, path where to find the weights.
        '''
        self.model.load_weights(path_weights)

    #def evaluate
    
    def get_classifier(self):
        exit()

#train_csv = ['dataset_csv_path/0&0s7', 'dataset_csv_path/0&1s7p0.1']
train_csv = ['dataset_csv_path/9-s7', 'dataset_csv_path/10-s7'] #'dataset_csv_path/1-s7p0.1'
dartDataHandler = DartDataHandler(train_csv, [], is_sources=[True, False])
mdm = MyDartModel(10, dartDataHandler, 25, 1, 10, with_kronecker=False)
mdm.train(checkpoint_path = 'Checkpoints/Dart_mnist_2', path_logger = 'csvLogger/Dart_mnist_2.csv')
mdm.save('weights/Dart_mnist_2', 'models/Dart_mnist_2')


'''train_csv = ['dataset_csv_path/0-s7', 'dataset_csv_path/1-s7']
dartDataHandler = DartDataHandler(train_csv, [ 'dataset_csv_path/7-s7'], is_sources=[True, False]) #'dataset_csv_path/7-s7' #'dataset_csv_path/0-s7',

myDartModel = MyDartModel(25, dartDataHandler, 25, 2, 10, with_kronecker=False,testing=True)
myDartModel.load_weights('Checkpoints/Dart_google_11')#'weights/Dart_la_muse_6')weights/Dart_la_muse_10
conf_matrix = myDartModel.confusion_matrix()

from plot import *
err = compute_error_per_label(np.array(dartDataHandler.get_labels()),conf_matrix)'''

