#!/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tf/bin/python3.9

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
import gc
import random as rn
import os

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, ActivityRegularization, Input
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

#self-defined function
exec(open("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/Deepsurv/trial_code_current/deepsurv_and_ML/tutorials/function_deepsurv.py").read())

# set random seed
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)
rn.seed(SEED)
os.environ['PYTHONHASHSEED'] = '0'

# cv1
# first X variables
# training data
train_x = pd.read_csv("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/data_10fold/individualCpGs/traincv1_x_alldeath.csv")
test_x = pd.read_csv("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/data_10fold/individualCpGs/testcv1_x_alldeath.csv")

# first select PI clinics
PIags_train =  train_x['PIags'].values
PIags_test =  test_x['PIags'].values

# then select methylation only, select only 100 methylation to try
train_cpgs = train_x.iloc[:, 4:]
test_cpgs = test_x.iloc[:, 4:]

# Standardization 
X_scaler = StandardScaler().fit(train_cpgs)
X_train = X_scaler.transform(train_cpgs)
X_test = X_scaler.transform(test_cpgs)


train_evtime = pd.read_csv("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/data_10fold/individualCpGs/traincv1_tdeath.csv")
test_evtime = pd.read_csv("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/data_10fold/individualCpGs/testcv1_tdeath.csv")

train_y = np.stack([train_evtime['timey'].values, 
                   train_evtime['death_all'].values], axis=1).astype('float64')

test_y = np.stack([test_evtime['timey'].values, 
                   test_evtime['death_all'].values], axis=1).astype('float64')

n_patients_train = X_train.shape[0]
n_features = X_train.shape[1]

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, train_y, PIags_train))
train_dataset = train_dataset.shuffle(buffer_size=n_patients_train).batch(batch_size, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, test_y, PIags_test))
val_dataset = val_dataset.batch(batch_size)

input_shape = n_features
# autoencoder pre-training
#encoder
inputs = Input(shape=(input_shape,))
latent = keras.layers.Dense(input_shape//2, activation='relu')(inputs)
latent = keras.layers.Dropout(0.5)(latent)
latent = keras.layers.BatchNormalization()(latent)
encoder = keras.Model(inputs, latent)

#decoder
outputs = keras.layers.Dense(input_shape, activation='sigmoid')(latent)
autoencoder = keras.Model(inputs, outputs)
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                    loss='binary_crossentropy')
                    
                    
### add random noise to the X_train
noise_factor = 0.5
X_train_noisy =  X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy =  X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

autoencoder.fit(X_train_noisy, X_train,
                epochs=100,
                 batch_size=32, validation_data= (X_test_noisy, X_test), 
                 callbacks =[tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, min_delta=0.0001, mode="min", patience=3)],
                 shuffle=False)
                                  
encoder_copy = keras.models.clone_model(encoder)
encoder_copy.set_weights(encoder.get_weights())

# build the deepsurv architecture
model = tf.keras.Sequential()
model.add(encoder_copy)
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(Dense(units=1, activation='linear', 
                     kernel_initializer='glorot_uniform', activity_regularizer=l2(0.001)))

print(model.summary())
# define hyperparameters
batch_size = 32
lr = 0.00001
epochs = 100
model_save_path = '/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/model/H1cv1'

early_stopping = EarlyStopping(model_path=model_save_path, patience=10, verbose=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

train_cpg = X_train
test_cpg = X_test

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    batches = 0
    for x_batch_train, y_batch_train, PIags in train_dataset:
        batches += 1
        if batches >= train_cpg.shape[0] // batch_size:
            break
        with tf.GradientTape() as tape:

            PIags = tf.cast(PIags, dtype=tf.float32)
            
            y_pred = model(x_batch_train, training=True)
            
            loss_value = cox_loss(y_batch_train, y_pred, PIags)
            
        grads = tape.gradient(loss_value, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_score = 0.03*(model.predict(train_cpg)) + 0.97*(PIags_train.reshape(PIags_train.shape[0], 1))
    test_score = 0.03*(model.predict(test_cpg)) + 0.97*(PIags_test.reshape(PIags_test.shape[0], 1))

    os_ci_train = concordance_index(train_y, -train_score)
    os_ci_test = concordance_index(test_y, -test_score)

    print(f'Test OS cindex:{os_ci_test}')

    early_stopping(-os_ci_test, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break      

train_score = 0.03*(model.predict(X_train)) + 0.97*(PIags_train.reshape(PIags_train.shape[0], 1))
test_score = 0.03*(model.predict(X_test)) + 0.97*(PIags_test.reshape(PIags_test.shape[0], 1))

os_ci_train = concordance_index(train_y, -train_score)
os_ci_test = concordance_index(test_y, -test_score)

print(f"c-index of training datasetcv1 = {os_ci_train}")
print(f"c-index of testing datasecv1 = {os_ci_test}")

# C-index
C_index = { 'num_cv' : [1],
           'num_hidden_layer' : [1],
           'C_train' : [os_ci_train.numpy()],
           'C_test' : [os_ci_test.numpy()]}
C_index = pd.DataFrame(C_index)
C_index.to_csv('/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/C_index/H1_cv1dl.csv')

