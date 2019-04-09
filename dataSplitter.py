import myUtils as mu
import os
import random
import shutil
import pandas as pd

class DataSplitter:

    def __init__(self, path_input, path_output, learn_prop = 0.7, test_prop = 0.2, valid_prop = 0.1, seed = 7):
                        
        self.learn_prop = learn_prop
        self.test_prop = test_prop
        self.valid_prop = valid_prop

        self.path_input = path_input
        self.path_output = path_output

        self.seed = seed

    def split(self):
        '''
            Split randomly the dataset into three sets (training, testing and validation set). The proportion of
            the three sets are respectively self.learn_prop, self.test_prop and self.valid_prop.
           
        '''
        self.labels = set(os.listdir(self.path_input)) 
        header = [True, True, True]
        for label in self.labels:
            # Retrieve the image names corresponding to label. 
            img_names = os.listdir(self.path_input + '/' + label)
            
            # To ensure the random splitting
            random.seed(self.seed),
            random.shuffle(img_names)
            l = len(img_names)    

            l = l
            last_index_train = int(l * self.learn_prop)
            last_index_test = last_index_train + int(l * self.test_prop)
            last_index_valid = last_index_test + int(l * self.valid_prop)

            for i, csv_name in enumerate(['_test', '_train', '_valid']):
                # get the random subpart of the img_name corresponding to csv_name (i.e: train, test or validation)
                if csv_name == '_train':
                    sub_img_names = img_names[0:last_index_train]
                elif csv_name == '_test':
                    sub_img_names = img_names[last_index_train:last_index_test]
                else:
                    sub_img_names = img_names[last_index_test:last_index_valid]

                data_to_write = {
                    'img_path': [self.path_input + '/' + label + '/' + img_name for img_name in sub_img_names],
                    'label': ['\'' + label + '\'' for i in range(len(sub_img_names))]
                }

                df = pd.DataFrame(data=data_to_write)
                df.to_csv(self.path_output + csv_name + '.csv',  mode='a', header=header[i])
                header[i] = False # The header have to be putted only once.


#DataSplitter('open_image_one_crop_by_img', 'croped_open_img_subset').split()
#DataSplitter('croped_images\images', 'croped_images\\lol').split()
#DataSplitter('croped_images\images', 'dataset_csv_path/lol2').split()
'''path_input = 'TESTCROP/7/valid_shipwreck'
labels = set(os.listdir(path_input))
first = True
for label in labels:
    img_names = os.listdir(path_input + '/' + label)
    data_to_write = {
        'img_path': ['TESTCROP/7/valid_shipwreck' + label + '/' + img_name for img_name in img_names],
        'label': [str(label) for i in range(len(img_names))]
    }
    
    df = pd.DataFrame(data=data_to_write)
    df.to_csv('dataset_csv_path/0&2s7p0.1_valid.csv', mode='a', header=first)
    first=False
    #break'''''