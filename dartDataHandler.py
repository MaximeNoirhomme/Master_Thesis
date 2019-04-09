from dataHandler import DataHandler, TRAIN_TYPE, TESTING_TYPE, VALIDATION_TYPE
import numpy as np
import myUtils as mu
class DartDataHandler(DataHandler):
    def __init__(self, train_csv_paths, test_csv_paths, is_sources = [True], batch_size = 32, mapping = None, seed = 7, width = 28, height = 28, 
                    nb_chanels = 1): # 224,224,3

        '''
            Constructor of DartDataHandler
            Parameters:
                - train_csv_paths: list of strings, names of the csv (without the _train.csv part) of the csv file that contains the training/validation splitting 
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
        
        super(DartDataHandler, self).__init__(train_csv_paths, test_csv_paths, batch_size, mapping, seed, width,
                                    height, nb_chanels)
        
        l = len(self.train_csv_paths)
        try:
            self.is_sources = mu.if_scalar_convert_to_list(is_sources, l, name='is_sources')
        except ValueError as e:
            raise ValueError(str(e).replace('l,', 'the variable paths'))

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
            domain_labels = np.empty((0, 1))

            for gen, is_source in zip(gens, self.is_sources):
                X1i = gen.next()

                x = np.concatenate((X1i[0], x))
                y = np.concatenate((X1i[1], y))

                if is_source: # if the domain is the source one, put a 0
                    domain_labels = np.concatenate((np.zeros((X1i[0].shape[0],1), dtype=np.float), domain_labels))      
                else: # otherwise, put a 1
                    domain_labels = np.concatenate((np.ones((X1i[0].shape[0], 1), dtype=np.float), domain_labels))

            x = [x, y, domain_labels]
            y = {
                'lab_class':y,
                'dom_class':domain_labels
            }    

            yield [x, y]


#DartDataHandler('TESTCROP/7').generator(TRAIN_TYPE)