from dataHandler import DataHandler, TRAIN_TYPE, TESTING_TYPE, VALIDATION_TYPE
import numpy as np
import myUtils as mu
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import pandas as pd
def u_shuffle(a,b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

LABEL = ['2206', '2231', '2451', '2805', '2807', '2861', '2879', '2912', '3062', '3102', '3224', '3265', '3509', '3564', '3731', '3786', '3827', '3859', '3866', '3875', '4318', '4341', '4350', '4455', '4479']
#LABEL = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#LABEL = ['2231', '2451', '2879', '3102', '3224', '3265', '3509', '3564', '3786', '3859', '3866', '4455', '4479']
LABEL = ['3564', '0']
LABEL = ['\'' + l + '\'' for l in LABEL]

class DartDataHandler(DataHandler):
    def __init__(self, train_csv_paths, test_csv_paths, is_sources = [0], batch_size = 32, mapping = None, seed = 7, width = 224, height = 224, 
                    nb_chanels = 3): # 28,28,1 #224,224,3

        '''
            Constructor of DartDataHandler
            Parameters:
                - train_csv_paths: list of strings, names of the csv (without the _train.csv part) of the csv file that contains the training/vali#2dation splitting 
                - test_csv_paths: list of string, names of the csv (without the _train.csv part) of the csv file that contains the testing splitting
                - is_sources: list of boolean, if the i^th element is True, then the i^th training dataset is source.
                - batch_size: Integer, batch size.
                - mapping: If set to something, it allows to change virtually the name of the labels (without changing the name of the folders).
                The mapping maps from real (given by folder names) to virtual label, and the label returns by "get_label(self)" are the virtual ones.
                - seed: Integer, seed
                - width: Integer, width of images
                - height: Integer, height of images
                - nb_chanels: Integer, nb of chanels of the images.
        '''
        self.nb_source = sum(is_sources)
        self.nb_target = len(is_sources) - self.nb_source
        
        self.train_csv_paths = train_csv_paths
        l = len(self.train_csv_paths)
        try:
            self.is_sources = mu.if_scalar_convert_to_list(is_sources, l, name='is_sources')
        except ValueError as e:
            raise ValueError(str(e).replace('l,', 'the variable paths'))

        super(DartDataHandler, self).__init__(train_csv_paths, test_csv_paths, batch_size, mapping, seed, width,
                                    height, nb_chanels)



    def _gen(self, data_type):
        gens = []
        it_batch_size_source = 0
        it_batch_size_target = 0

        last_source = 0
        last_target = 0
        for j, is_source in enumerate(self.is_sources):
            if is_source:
                last_source = j
            else:
                last_target = j

        for i, (csv_path, is_source) in enumerate(zip(self.all_paths[data_type], self.is_sources)):
            # The data augmentation is only needed for the learning phase.
            if data_type == TRAIN_TYPE:
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
                datagen = ImageDataGenerator(rescale=1./255)#
            
            if data_type == TESTING_TYPE:
                batch_size = 1
            else:
                if i == last_source:#len(self.all_paths[data_type]) - 1:
                    #if is_source:
                    batch_size = int(self.batch_size/2) - it_batch_size_source
                    #else:
                    #    batch_size = self.batch_size - it_batch_size_target
                elif i == last_target:
                    batch_size = int(self.batch_size/2) - it_batch_size_target 
                else:
                    if is_source:
                        batch_size = int(self.batch_size / (2*self.nb_source))
                        it_batch_size_source += batch_size
                    else:
                        batch_size = int(self.batch_size / (2*self.nb_target))
                        it_batch_size_target += batch_size

            # Create a dataframe from csv
            df=pd.read_csv(csv_path)

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
            domain_labels = np.zeros((0, 1), dtype=np.int32)
            
            for gen, is_source in zip(gens, self.is_sources):
                X1i = gen.next()
                x = np.concatenate((X1i[0], x))
                y = np.concatenate((X1i[1], y))

                '''if is_source: # if the domain is the source one, put a 0
                    domain_labels = np.concatenate((np.zeros((X1i[0].shape[0],1), dtype=np.int32), domain_labels))      
                else: # otherwise, put a 1
                    domain_labels = np.concatenate((np.ones((X1i[0].shape[0], 1), dtype=np.int32), domain_labels))'''

                domain_labels = np.concatenate((is_source*np.ones((X1i[0].shape[0], 1), dtype=np.int32), domain_labels))        
            
            domain_labels_hot_encoded = np.zeros((domain_labels.shape[0], 2))
            domain_labels_hot_encoded[np.arange(domain_labels.shape[0]), domain_labels[:,0]] = 1

            x = [x, y, domain_labels]
            y = {
                'lab_class':y,
                'dom_class':domain_labels_hot_encoded
            }    

            yield [x, y]


#DartDataHandler('TESTCROP/7').generator(TRAIN_TYPE)