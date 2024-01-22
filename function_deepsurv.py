"""
Early stopping object

from https://github.com/Bjarten/early-stopping-pytorch

"""

import numpy as np
import tensorflow as tf

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model_path, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_path = model_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...{}'.format(self.val_loss_min, val_loss, self.model_path))
        # torch.save(model.state_dict(), self.model_path)
        model.save(self.model_path)
        self.val_loss_min = val_loss
                
## define loss function
def cox_loss(y_true, y_pred, PIags):
    time_value = tf.squeeze(tf.gather(y_true, [0], axis=1))
    event = tf.cast(tf.squeeze(tf.gather(y_true, [1], axis=1)), tf.bool)
    
    score_m = tf.squeeze(y_pred, axis=1)
    score = tf.add(0.3*score_m, 0.7*PIags) # add two h(x), with relative weight

    ix = tf.where(event)

    sel_mat = tf.cast(tf.gather(time_value, ix) <= time_value, tf.float32)

    p_lik = tf.gather(score, ix) - tf.math.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))

    loss = -tf.reduce_mean(p_lik)

    return loss


# define C index

def concordance_index(y_true, y_pred):
    time_value = tf.squeeze(tf.gather(y_true, [0], axis=1))
    event = tf.cast(tf.squeeze(tf.gather(y_true, [1], axis=1)), tf.bool)
    ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
    ix = tf.where(tf.logical_and(tf.expand_dims(time_value, axis=-1) < time_value,
                                 tf.expand_dims(event, axis=-1)), name='ix')

    ## count how many score[i]<score[j]
    s1 = tf.gather(y_pred, ix[:, 0])
    s2 = tf.gather(y_pred, ix[:, 1])
    ci = tf.reduce_mean(tf.cast(s1 < s2, tf.float32), name='c_index')

    return ci

