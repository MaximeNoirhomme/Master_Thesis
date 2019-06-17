from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import pickle
import numpy as np
import pandas as pd
import csv 

class YoloDataHandler:
    def __init__(self, csv_path, batch_size = 32, width=224, height=224, nb_chanels=3, size_bb=11*11*2*4):
        self.csv_path = csv_path
        self.batch_size = batch_size
        
        self.width = width
        self.height = height
        self.color_mode = 'grayscale' if nb_chanels == 1 else 'rgb'
        self.size_bb = size_bb

        self.gen, self.mapping = self._gen()

    def _gen(self):
        datagen = ImageDataGenerator(rescale=1./255)

        df=pd.read_csv(self.csv_path)

        gen = datagen.flow_from_dataframe(dataframe=df,
                                directory=None,
                                x_col="img_path",
                                y_col=["id"],
                                class_mode="other",
                                target_size=(self.width, self.height),
                                batch_size=self.batch_size,
                                color_mode=self.color_mode)

        mapping = {}
        with open(self.csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tmp = np.load(row['label'])
                mapping[row['id']] = tmp

        return gen, mapping

    def generator(self):     
        while True:
            y = None
            X1i = self.gen.next()
            x = X1i[0]
            ids = X1i[1]

            y = [self.mapping[str(ids[i,0])] for i in range(ids.shape[0])]

                    
            x = [x]
            y = [y]  
 
            yield [x, y]
