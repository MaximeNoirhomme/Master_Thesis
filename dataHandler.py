import os
import myUtils as mu
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from subprocess import call
import numpy as np
from numpy.random import seed as s
import pandas as pd
from collections import OrderedDict
import csv
# Constantes definition:

TRAIN_TYPE = 0
TESTING_TYPE = 1
VALIDATION_TYPE = 2

LABEL = ['2206', '2231', '2451', '2805', '2807', '2861', '2879', '2912', '3062', '3102', '3224', '3265', '3509', '3564', '3731', '3786', '3827', '3859', '3866', '3875', '4318', '4341', '4350', '4455', '4479']
LABEL = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LABEL = ['\'' + l + '\'' for l in LABEL]
class DataHandler:
    def __init__(self, train_csv_paths, test_csv_paths, batch_size = 32, mapping = None, seed = 7, width = 224, height = 224, 
                        nb_chanels = 3):
        '''
            Constructor of DataHandler.
            Parameters:
                - train_csv_paths: list of strings, names of the csv (without the _train.csv part) of the csv file that contains the training/validation splitting 
                - test_csv_paths: list of string, names of the csv (without the _train.csv part) of the csv file that contains the testing splitting
                - batch_size: Integer, batch size.
                - mapping: If set to something, it allows to change virtually the name of the labels (without changing the name of the folders).
                The mapping maps from real (given by folder names) to virtual label, and the label returns by "get_label(self)" are the virtual ones.
                - seed: Integer, seed
                - width: Integer, width of images
                - height: Integer, height of images
                - nb_chanels: Integer, nb of chanels of the images.
        '''

        print(seed)
        s(seed)        
        
        self.width = width
        self.height = height
        self.nb_chanels = nb_chanels
        self.color_mode = 'grayscale' if nb_chanels == 1 else 'rgb'
        print(self.color_mode)

        self.train_csv_paths = [path + '_train.csv' for path in train_csv_paths]
        self.test_csv_paths = [path + '_test.csv' for path in test_csv_paths]
        self.valid_csv_paths = [path + '_valid.csv' for path in train_csv_paths]

        self.all_paths = [self.train_csv_paths, self.test_csv_paths, self.valid_csv_paths]

        print(self.all_paths)

        self.batch_size = batch_size
        self.mapping = mapping
        self.seed = seed
        self._size = [None, None, None]
        self._labels = [None, None, None]

        self._gen_train = None
        self._gen_test = None
        self._gen_val = None
    
    def gen_train(self):
        if self._gen_train == None:
            self._gen_train = self._gen(TRAIN_TYPE)
        return self._gen_train

    def gen_test(self):
        if self._gen_test == None:
            self._gen_test = self._gen(TESTING_TYPE)
        return self._gen_test

    def gen_val(self):
        if self._gen_val == None:
            self._gen_val = self._gen(VALIDATION_TYPE)
        return self._gen_val        

    def get_size(self, data_type):
        '''
            Returns the size of the part of the dataset that corresponds to data_type. For example
            get_size(TRAIN_TYPE) will return the size of the learning dataset.
            Notice that, the getter is a lazy loader.
            Parameters:
                - data_type: An integer, the type of the data (training (0), testing (1) or validation type (2)).
        '''
        self._nb_classes = []
        if self._size[data_type] == None:
            self._size[data_type] = 0

            for csv_path in self.all_paths[data_type]:
                self._size[data_type] += mu.get_size_csv(csv_path)
                                    
        return self._size[data_type]

    def get_balanced_weights(self, data_type):
        '''
            Compute weights in order to balance dataset with unbalanced number of sample per classes.
            The weights are computed as: total_size / (nb_sample(i) * nb_classes), where total_size is the
            total size of the dataset, nb_sample(i), the number of sample for the class i and nb_classes, the number
            of classes in the dataset.
            Parameters:
                - data_type: An integer, the type of the data (training (0), testing (1) or validation type (2)).
            Return:
                An numpy array of size nb_classes.
        '''
        nb_classes = self.get_nb_classes(data_type)
        weights = np.empty((nb_classes))
        total_size = self.get_size(data_type)
        for i, label in enumerate(self.get_labels()):
            weights[i] = total_size / (self.get_size_label(data_type, label) * nb_classes) 

        print(weights)

        return weights

    def get_labels(self):
        return LABEL

    def get_nb_classes(self, datatype): #TODO: compute nb_classes in a dynamic way ?
        '''
            Return the number of classes in the dataset.
            Notice that the getter is a lazy loader.
        '''
        return len(self.get_labels())

    def get_size_label(self, data_type, label):
        '''
            Returns the size of the part of the dataset that corresponds to data_type and label. For example
            get_size(TRAIN_TYPE, '2000') will return the number of learning sample labelled as '2000'.
            Parameters:
                - data_type: An integer, the type of the data (training (0), testing (1) or validation type (2)).
                - label: A string, the label corresponding to sample to count.
        '''
        size = 0
        for csv_path in self.all_paths[data_type]:
            with open(csv_path) as csv_file:
                size += sum([1 if row['label'] == label else 0 for row in csv.DictReader(csv_file)])
                    
        return size 

    def _gen(self, data_type):
        gens = []
        
        it_batch_size = 0

        for i, csv_path in enumerate(self.all_paths[data_type]):
            # The data augmentation is only needed for the learning phase.
            if data_type == TRAIN_TYPE:
                print('train')
                datagen = ImageDataGenerator(
                                rescale=1./255,
                                horizontal_flip = True,
                                vertical_flip = False,
                                height_shift_range = 0.15,
                                width_shift_range = 0.15,
                                rotation_range = 5,
                                shear_range = 0.01,
                                fill_mode = 'nearest',
                                zoom_range=0.25)
            else:
                datagen = ImageDataGenerator(rescale=1./255)
            
            if data_type == TESTING_TYPE:
                batch_size = 1
            else:

                if i >= len(self.all_paths[data_type]) - 1:
                    batch_size = self.batch_size - it_batch_size
                else:
                    batch_size = int(self.batch_size / len(self.all_paths[data_type]))
                    it_batch_size += batch_size
                
            # Create a dataframe from csv
            df=pd.read_csv(csv_path)

            print("batch_size = ", batch_size)
            # define the generator thanks to the dataframe.
            gen = datagen.flow_from_dataframe(dataframe=df,
                        directory=None,
                        x_col="img_path",
                        y_col="label",
                        class_mode="categorical",
                        target_size=(self.width, self.height),
                        batch_size=batch_size,
                        classes=LABEL,
                        color_mode=self.color_mode)
            
            gens.append(gen)
        
        return gens
    
    def generator(self, data_type):
        '''
            Define generators.
            Parameters:
                - data_type: Integer, indicates which generator is requested (the one for training, validation or testing).
        '''
        
        gens = self.gen_train() if data_type == TRAIN_TYPE else (self.gen_test() if data_type == TESTING_TYPE else self.gen_val())

        while True:
            x = np.empty((0, self.width, self.height, self.nb_chanels))
            y = np.empty((0, self.get_nb_classes(data_type)))
            for gen in gens:
                X1i = gen.next()

                x = np.concatenate((X1i[0], x))
                y = np.concatenate((X1i[1], y))

            x = [x]
            y = [y]    

            yield [x, y]