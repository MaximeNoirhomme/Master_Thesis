'''
    This file does not belong to me.
    Find the source file at 
    'https://github.com/paintception/Deep-Transfer-Learning-for-Art-Classification-Problems/blob/master/saliency_maps_activations/visual_backprop.py'
'''
from saliency import SaliencyMask
import numpy as np
import keras.backend as K
from keras.layers import Input, Conv2DTranspose, ZeroPadding2D
from keras.models import Model
from keras.initializers import Ones, Zeros

class VisualBackprop(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with VisualBackprop (https://arxiv.org/abs/1611.05418).
    """

    def __init__(self, model, arch_type='vgg19', output_index=0):
        """Constructs a VisualProp SaliencyMask."""
        self.arch_type = arch_type

        if self.arch_type == 'vgg19':
            self.cond = lambda x, y, i : 'Conv2D' in str(type(x))
            self.comp = self._vgg_comp
        elif self.arch_type == 'resnet50':
            self.cond = lambda x, y, i : 'Activation' in str(type(x)) and len(y[i].shape) == 4
            self.comp = self._resnet_comp
        else:
            raise ValueError('The visualisation is only supported for vgg19 and resnet50 and get ' + arch_type)
        
        inps = [model.input, K.learning_phase()]           # input placeholder
        outs = [layer.output for layer in model.layers]    # all layer outputs
        outs = outs[1:]
        
        self.forward_pass = K.function(inps, outs)         # evaluation function

        self.model = model
        self.model.dssqfdqs = 2
        self.model.dssqfdqs += 1 
        print(self.model.dssqfdqs)
    def get_mask(self, input_image, max_gg=0):
        """Returns a VisualBackprop mask."""
        x_value = np.expand_dims(input_image, axis=0)
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])

        l = False

        for i in range(len(self.model.layers)-1-1, -1, -1):
            if self.cond(self.model.layers[i], layer_outs, i):
                layer = self.comp(layer_outs[i])

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:                      
                        if layer.shape == (1,114,114,1):
                            l = True
                        
                        visual_bpr = self._deconv(visual_bpr)
                        
                        if l:          
                            visual_bpr = self._padding(visual_bpr, (1,1))
                            l=False

                    visual_bpr = visual_bpr * layer
                    
                else:
                    visual_bpr = layer

        return visual_bpr[0]
    
    def _deconv(self, feature_map):
        """The deconvolution operation to upsample the average feature map downstream"""

        x = Input(shape=(None, None, 1))

        y = Conv2DTranspose(filters=1, 
                            kernel_size=(3,3),
                            strides=(2,2), 
                            padding='same',
                            kernel_initializer=Ones(), 
                            bias_initializer=Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input, K.learning_phase()]   # input placeholder                                
        outs = [deconv_model.layers[-1].output]           # output placeholder
        deconv_func = K.function(inps, outs)              # evaluation function
        
        return deconv_func([feature_map, 0])[0]

    def _padding(self, feature_map, padding):
        """The padding operation"""
        x = Input(shape=(None, None, 1))
        y = ZeroPadding2D(padding=padding)(x)

        padding_model = Model(inputs=[x], outputs=[y])
        
        inps = [padding_model.input, K.learning_phase()]                                   
        outs = [padding_model.layers[-1].output]           
        padding_func = K.function(inps, outs)    
        
        return padding_func([feature_map, 0])[0]

    def _vgg_comp(self, output):
        """Computation of mean according to kernel dimension for vgg"""
        layer = np.mean(output, axis=3, keepdims=True)
        layer = layer - np.min(layer)
        return layer/(np.max(layer)-np.min(layer)+1e-6)

    def _resnet_comp(self, output):
        """Computation of mean according to kernel dimension for resnet"""
        layer = np.mean(output, axis=3, keepdims=True)
        #return layer
        layer = layer + np.min(abs(layer))
        return layer/(np.max(layer)-np.min(layer)+1e-6)


'''from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img, img_to_array
from plot import plot_cmp_img 
model = ResNet50('imagenet')
print(model.summary())
#model = VGG19('imagenet')
v = VisualBackprop(model, arch_type='resnet50')

im = load_img('individualImage.jpg', target_size=(224,224))
img = np.asarray(im)#img_to_array(img)
x = np.expand_dims(img, axis=0)

masks = [v.get_mask(x[0], i) for i in range(1)] 
plot_cmp_img('a', [im] + masks, [str(i) for i in range(2)], output_path=None)'''