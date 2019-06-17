from keras.callbacks import Callback
import keras.backend as K
class dartParamUpdate(Callback):
    def __init__(self, nb_epoch, step_per_epoch, mu0, alpha, beta):
        self.nb_epoch = nb_epoch
        self.step_per_epoch = step_per_epoch
        self.total_it = nb_epoch * step_per_epoch
        self.mu0 = mu0
        self.alpha = alpha
        self.beta = beta
        self.it = 0

    def on_batch_end(self, batch, lorgs={}):
        self.it += 1
        lr = self.mu0 / (1 + self.alpha * (self.it / self.total_it)) ** self.beta
        
        if self.it % 500 == 0:
            print("lr = ", lr, " ")
            
        K.set_value(self.model.optimizer.lr, lr)
        flip_grad_layer = self.model.get_layer('gradient_reversal_1')
        incr = 1. / self.total_it
        flip_grad_layer.update_lambda(incr)