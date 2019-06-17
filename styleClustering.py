from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import AveragePooling2D
import keras.backend as K
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import DBSCAN
import os
import tensorflow as tf
import myUtils as mu
import shutil
import pickle

INDICES = [17] #[1, 4, 7, 12, 17]
class StyleClustering:
    def __init__(self, imgs_path):
        self.imgs_path = imgs_path
        self.model = self._get_model()
        self.forward_pass = self._get_forward_pass()

    def cluster(self, metric_paths = None):
        # Load images, extract features and compute grams matrix.
        grams = []
        names = []
        i = 0
        for img_name in os.listdir(self.imgs_path):
            names.append(img_name)
        
        if metric_paths == None:
            for img_name in os.listdir(self.imgs_path):
                img = load_img(self.imgs_path + '/' + img_name, target_size=(224,224))
                img = np.asarray(img)
                img = np.expand_dims(img, axis=0)
                
                features = self.extract_features(img)
                gram = [self.compute_gram(feature) for feature in features]
                
                names.append(img_name)
                grams.append(gram)

            metrics = np.zeros((len(grams), len(grams)))
            for i in range(len(grams)):
                for j in range(i + 1):
                    metrics[i][j] = self.style_metrix(grams[i], grams[j])
                    metrics[j][i] = metrics[i][j]

            np.save("metric2.npy", metrics)
        else:
            metrics = np.load("metric2.npy")
        #
        dbscan = DBSCAN(metric='precomputed', min_samples=5, eps=10**3).fit_predict(metrics)
        u = np.unique(dbscan)
        dic = {}
        for a in u:
            if a in dic:
                continue
            else:
                dic[a] = 0
        
        for i in dbscan:
            dic[i] += 1

        mu.create_folder_if_not_exist("Cluster")
        for i, img_name in enumerate(os.listdir(self.imgs_path)):
            mu.create_folder_if_not_exist('Cluster/'+str(dbscan[i]))
            shutil.copyfile(self.imgs_path + '/' + img_name, 'Cluster/'+str(dbscan[i]) + '/' + img_name)

        print(dic)
    
    def TSNE(self):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns 
        sns.set()
            
        metrics = np.load("metricCool.npy")

        X_embedded2 = TSNE(n_components=2, perplexity=5).fit_transform(metrics)

        pickle_in = open("D:/Users/Noirh/Documents/TFE/figure/TSNE/style-dic", "rb")
        img_dic = pickle.load(pickle_in)

        x1 = mu.plot_embedding(X_embedded2, 1869, ['Cytomine_dataset/' + a for a in os.listdir('Cytomine_dataset')], img_dic=img_dic)#dic_name='D:/Users/Noirh/Documents/TFE/figure/TSNE/style-dic')
            
        if x1 != None:
            X_embedded2 = x1

        for n in img_dic:
            x = img_dic[n]
            if x[0] >= 0.375245 and x[0] <= 0.391175 and x[1] <= 0.96762 and x[1] >= 0.901288:
                print(n)
        plt.show()
        for i in range(1869):
            plt.scatter(X_embedded2[i][0], X_embedded2[i][1],c="r", s=1)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        


    def style_metrix(self, grams1, grams2):
        '''
            Define the similarity between two GramObject grams1 and grams2 according to their style.
            The Similarity is defined as the mean-squared distance between the entries of the
            Gram matrix as define in https://arxiv.org/pdf/1505.07376.pdf.
            Parameters:
                -grams1: a list of 2-dimentional numpy array
                -grams2: a list of 2-dimentional numpy array
            Return:
                - A float number, the computate distance.
        '''
        dist = 0
        for i in range(len(grams1)):
            gram1 = grams1[i]
            gram2 = grams2[i]

            dist += np.sum(gram1 - gram2)**2
        
        #print(dist)
        return dist

    def extract_features(self, img):       
        # Compute all outputs
        layer_outs = self.forward_pass([img, 0])
        # Only keep the relevant features and reshape them to the form (M_l, N_l) where M_l is
        # the size of a filter and N_l the number of filter.
        features = [np.reshape(layer_outs[i], (-1, layer_outs[i].shape[3])) for i in INDICES]

        return features

    def compute_gram(self, feature):
        return np.matmul(feature.transpose(), feature) / feature.size

    def _get_model(self):
        '''
            Get VGG19 model pretrained on imagenet where the Maxpooling layers have been replaced by
            averagepooling ones.
        '''

        model = VGG19(weights='imagenet')

        x = None
        for i in range(len(model.layers)):
            # if the layer is a Maxpooling, replace it by an averagepooling.
            if 'MaxPooling2D' in str(type(model.layers[i])):
                model.layers[i] = AveragePooling2D()

            x = model.layers[i].output if x == None else model.layers[i](x)
                    
        model = Model([model.layers[0].input], [x])

        # recompile the model (the optimizer and the loss does not matter since the model won't be trained again)
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def _get_forward_pass(self):
        # code inspired from visualization.py which is not my code.
        inps = [self.model.input, K.learning_phase()]           # input placeholder

        outs = []
        for layer in self.model.layers:
            try:
               outs.append(layer.get_output_at(1))
            except:
                outs.append(layer.output)
        
        outs = outs[1:] # all layer outputs except for input layer

        return K.function(inps, outs)

class GramObject:
    def __init__(self, grams):
        self.grams = grams

    def __getitem__(self, key):
        return self.grams[key]

StyleClustering('Cytomine_dataset').TSNE()#.cluster()       