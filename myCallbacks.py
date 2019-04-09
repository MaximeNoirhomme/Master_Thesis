from keras.callbacks import Callback

class dartParamUpdate(Callback):
    def __init__(self, nb_epoch):
        self.nb_epoch = nb_epoch

    def on_epoch_end(self, epoch, logs={}):
        flip_grad_layer = self.model.get_layer('gradient_reversal_1')#'GradientReversal')
        incr = 1 / self.nb_epoch
        flip_grad_layer.update_lambda(incr)