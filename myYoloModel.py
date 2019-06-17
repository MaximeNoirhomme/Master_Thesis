from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD, Adam
import myUtils as mu 
import tensorflow as tf 
from yoloDataHandler import YoloDataHandler
from myDartModel import MyDartModel
from dartDataHandler import DartDataHandler
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np 
from scipy.special import expit as sigmoid

class MyYoloModel:
    def __init__(self, trained_extractor, S, B, C, l_coord, l_noobj, prob_threshold, non_max_threshold, dataHandler):
        self.trained_extractor = trained_extractor
        self.S = S
        self.B = B
        self.C = C

        self.l_coord = l_coord
        self.l_noobj = l_noobj

        # Definition of the size of the bounding boxes information, class probability
        # and confidence tensor.
        self.size_bb_tensor = (-1, S * S * 4 * B)
        self.size_cp_tensor = (-1, S * S * C)
        self.size_conf_tensor = (-1, S * S * B)

        self.prob_threshold = prob_threshold
        self.non_max_threshold = non_max_threshold

        self.dataHandler = dataHandler

        self.model = self._get_model()

    def train(self, start_epoch = 0, nb_epoch = 25, early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto'), 
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
        
        step_per_epoch = mu.get_size_csv(self.dataHandler.csv_path) // self.dataHandler.batch_size
        callbacks = []
        if early_stopping != None:
            callbacks.append(early_stopping)

        if checkpoint_path != None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                        monitor='val_loss',#
                                        save_weights_only=True,
                                        verbose=1,
                                        save_best_only=True)
            
            callbacks.append(cp_callback)

        if path_logger != None:    
            csv_logger = tf.keras.callbacks.CSVLogger(path_logger)
            callbacks.append(csv_logger)

        history = self.model.fit_generator(self.dataHandler.generator(), steps_per_epoch=step_per_epoch,
			epochs=nb_epoch, verbose=1, callbacks=callbacks, class_weight=None, max_queue_size=10, workers=1,
			 use_multiprocessing=False, shuffle=True, initial_epoch=start_epoch)
       
        return history
    
    
    def _get_model(self):
        inp_yolo = self.trained_extractor.input
        out_ext = self.trained_extractor.output
        # Definition of the yolo part.
        x = Dense(units=4096)(out_ext)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)
        x = Dense(units=self.S*self.S*(5*self.B + self.C), activation='linear')(x)


        yolo_model = Model([inp_yolo], [x])
        yolo_model.compile(optimizer=Adam(lr=0.001, decay=0.0001), loss=self._yolo_loss(), metrics=[self.debug()]) #SGD(lr = 0.001, momentum = 0.9) Adam(lr=0.001)

        return yolo_model
    def _yolo_loss(self):
        def loss(y_true, y_pred):
            # First, we have to extract bounding boxes information, class probability and confidence
            # from the true and predicted label

            # bounding boxes information (i.e: x,y,w,h):
            bb_true = K.reshape(K.slice(y_true, (0,0), self.size_bb_tensor), (-1, self.S, self.S, self.B, 4))
            bb_pred = K.slice(y_pred, (0,0), self.size_bb_tensor)
            bb_pred = tf.sigmoid(bb_pred)
            bb_pred = K.reshape(bb_pred, (-1, self.S, self.S, self.B, 4))

            # Confidence of bounding boxes
            conf_true = K.reshape(K.slice(y_true, (0, self.size_bb_tensor[1]), self.size_conf_tensor),  (-1, self.S, self.S, self.B))
            conf_pred = K.reshape(K.slice(y_pred, (0, self.size_bb_tensor[1]), self.size_conf_tensor),  (-1, self.S, self.S, self.B))
            
            # class probability
            cp_true = K.reshape(K.slice(y_true, (0, self.size_bb_tensor[1] + self.size_conf_tensor[1]), self.size_cp_tensor), (-1, self.S, self.S, self.C))
            cp_pred = K.reshape(K.slice(y_pred, (0, self.size_bb_tensor[1] + self.size_conf_tensor[1]), self.size_cp_tensor), (-1, self.S, self.S, self.C))
            
            # Computation of IOU

            # Compute the intersection between the two bounding boxes and the union
            # First compute the difference of the centers
            x_dif = K.abs(bb_true[:,:,:,:,0] - bb_pred[:,:,:,:,0] * 224./11.)
            y_dif = K.abs(bb_true[:,:,:,:,1] - bb_pred[:,:,:,:,1] * 224./11.)
            # Then compute the width and height of the common area

            w_inter = (bb_true[:,:,:,:,2] + bb_pred[:,:,:,:,2]*224) / 2 - x_dif
            h_inter = (bb_true[:,:,:,:,3] + bb_pred[:,:,:,:,3]*224) / 2 - y_dif
            
            # It can't be negative
            w_inter = K.tf.where(w_inter < 0, K.zeros(K.shape(w_inter)), w_inter)
            h_inter = K.tf.where(h_inter < 0, K.zeros(K.shape(h_inter)), h_inter)

            area_inter = w_inter * h_inter

            area_union = K.clip(bb_true[:,:,:,:,2] * bb_true[:,:,:,:,3] + \
                bb_pred[:,:,:,:,2] * bb_pred[:,:,:,:,3] * 224 * 224 - area_inter, 10**-4, None)

            iou = area_inter / area_union
            iou = K.clip(iou, 0,1)
            # Now compute the masks defined in the report.
            # The prob to have an object in a cell is the sum of all classes for that cell (sum P(class_i|object) = P(object))
            prob_object_non_reshape = K.clip(K.sum(cp_true, axis=-1),0,1)
            prob_object = K.reshape(prob_object_non_reshape, (-1, self.S, self.S, 1)) #conf_true

            conf_bb = iou * K.tile(prob_object, [1, 1, 1, self.B])
            conf_true = conf_bb
            # Get the index of the most probable bounding boxes among B for each cell.
            bb_index = K.argmax(conf_bb, axis=-1)

            mask_obj_i = prob_object_non_reshape#K.clip(K.sum(prob_object, axis=-1),0,1)
            mask_obj_ij = K.one_hot(bb_index, self.B)
            mask_noobj_ij = K.abs(mask_obj_ij - K.ones(K.shape(mask_obj_ij)))


            w_dif = K.sqrt(bb_true[:,:,:,:,2]) - K.sqrt(bb_pred[:,:,:,:,2]*224)
            h_dif = K.sqrt(bb_true[:,:,:,:,3]) - K.sqrt(bb_pred[:,:,:,:,3]*224)
            conf_dif_square = K.square(conf_true - conf_pred)
            cp_dif = K.square(cp_true - cp_pred)

            loss = K.clip(self.l_coord * K.sum(mask_obj_ij * (K.square(x_dif) + K.square(y_dif) + K.square(w_dif) + K.square(h_dif)), axis=[1,2,3]), 0, 10**9)
            loss += K.sum(mask_obj_ij * conf_dif_square, axis=[1,2,3])
            loss += self.l_noobj * K.sum(mask_noobj_ij * conf_dif_square, axis=[1,2,3]) 
            loss += K.sum(mask_obj_i * K.sum(cp_dif, axis=-1), axis=[1,2])

            return loss

        return loss

    def predict(self, img_path, bb_true):
        
        x = load_img(img_path, target_size=(224,224))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = x/255.

        output = self.model.predict(x)
        output = output[0]

        # bounding boxes information (i.e: x,y,w,h):
        bb_preds = np.reshape(output[0:self.size_bb_tensor[1]], (self.S, self.S, self.B, 4))
        bb_preds = sigmoid(bb_preds)

        bb_preds[:,:,:,0] *= 224./11.
        bb_preds[:,:,:,1] *= 224./11. 
        bb_preds[:,:,:,2] *= 224 
        bb_preds[:,:,:,3] *= 224

        # Confidence of bounding boxes
        conf_preds = np.reshape(output[self.size_bb_tensor[1]:self.size_bb_tensor[1] + self.size_conf_tensor[1]], (self.S, self.S, self.B))
        
        # class probability
        cp_preds = np.reshape(output[self.size_bb_tensor[1] + self.size_conf_tensor[1]:], (self.S, self.S, self.C))
         
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        for i in range(self.C):
            boxes = self._bb_filter_per_class(i, bb_preds, cp_preds, conf_preds)

            for box in boxes:
                x,y,w_,h_ = box.bb_pred[0],box.bb_pred[1],box.bb_pred[2],box.bb_pred[3]

                x1, x2 = x - w_/2, x + w_/2
                y2, y1 = y - h_/2, y + h_/2
                
                img = cv2.rectangle(img, (int(x1),224-int(y1)), (int(x2),224-int(y2)), (0,255,0),3)

        t_bb = np.reshape(bb_true, (self.S,self.S,self.B, 4))
        for l in range(self.S):
            for m in range(self.S):
                for p in range(self.B):
                    x,y,w_,h_ = t_bb[l,m,p,0] + l * 224 / 11, t_bb[l,m,p,1] + m * 224 / 11,t_bb[l,m,p,2],t_bb[l,m,p,3]
                    if w_ == 0.0 or h_ == 0.0:
                        continue
                    
                    x1, x2 = x - w_/2, x + w_/2
                    y2, y1 = y - h_/2, y + h_/2
                    
                    img = cv2.rectangle(img, (int(x1),224-int(y1)), (int(x2),224-int(y2)), (255,0,0),3)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _bb_filter_per_class(self, class_index, bb_preds, cp_preds, conf_preds):
        boxes = []
        # get boxes whose the bounding boxes probability is high enough
        for i in range(self.S):
            for j in range(self.S): 
                for k in range(self.B):
                    prob = cp_preds[i,j,class_index] * conf_preds[i,j,k]
                    
                    prob = 0 if prob < self.prob_threshold else prob
                    bb = bb_preds[i,j,k,:]

                    bb[0] += i*224/11
                    bb[1] += j*224/11

                    boxes.append(Bounding_boxes(bb, prob))

        # Apply non-maximal reduction algorithm
        boxes.sort(key=lambda box:box.prob, reverse=True)

        for i, box in enumerate(boxes):
            if box.prob <= 0:
                continue

            for j in range(i + 1, len(boxes)):
                if mu.iou_computation(box.bb_pred, boxes[j].bb_pred) > self.non_max_threshold:
                    box.prob = 0

        box = [box for box in boxes if box.prob > 0]

        return box

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

class Bounding_boxes:
    def __init__(self, bb_pred, prob):
        self.bb_pred = bb_pred
        self.prob = prob

'''ydh = YoloDataHandler('dataset_csv_path/yolo4-2.csv', batch_size=8)

train_csv = ['dataset_csv_path/11-s7_2', 'dataset_csv_path/8-s7_2']
dartDataHandler = DartDataHandler(train_csv, [], is_sources=[0,1])
mdm = MyDartModel(0, 2, dartDataHandler, 25, 2, 2, alpha=0.6, with_kronecker=False, complex_m=False)# 25
mdm.load_weights('weights/small_cyto')
dartModel = mdm.model

output = None
for layer in dartModel.layers:
    if layer.name == 'output_extractor':
        output = layer.output
        break

yolo_extractor = Model(dartModel.input[0], output)  

import sys
mym = MyYoloModel(yolo_extractor, 11, 2, 2, 5, 0.5, 0.1, 0.4, ydh)
mym.train(checkpoint_path = 'Checkpoints/yolo5-2', path_logger = 'csvLogger/yolo5-2.csv', nb_epoch=15)
mym.save('weights/yolo5-2', 'models/yolo5-2')
mym.load_weights('weights/yolo5-2')
import os
for f in os.listdir('D:/Users/Noirh/Documents/TFE/Yolodataset4-2'):
    name, ext = os.path.splitext('D:/Users/Noirh/Documents/TFE/Yolodataset4-2/' + f)
    if ext == '.png':
        a = np.load(name+'.npy')
        bb = np.reshape(a[:11*11*2*4], (11,11,2,4))
        conf = np.reshape(a[11*11*2*4:11*11*2*4 + 11*11*2], (11,11,2))
        prob = np.reshape(a[11*11*2*4 + 11*11*2:], (11,11,2))
        
        mym.predict(name+'.png', np.load(name+'.npy')[:11*11*2*4])'''
