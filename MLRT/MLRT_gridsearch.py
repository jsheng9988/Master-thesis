import numpy as np 
import os

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn import preprocessing

#######################################################################

class SaveModelEpoch(keras.callbacks.Callback):
    def __init__(self, filepath):
        super(SaveModelEpoch, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        epoch_str = '{:04d}'.format(epoch + 1)  # Format epoch number
        filepath = self.filepath.format(epoch=epoch_str)
        self.model.save(filepath)
        #print(f"\nSaved model at epoch {epoch + 1} to {filepath}")
    
#######################################################################

if __name__ == "__main__":

    DEBUG = True

    focks = np.load("td_Fock_at_re+im_rt-tdexx_delta_s0_h2_sto-3g_PYSCF.npz")
    denss = np.load("td_dens_re+im_rt-tdexx_delta_s0_h2_sto-3g_PYSCF.npz")

    totdim = focks['td_fock_at_re_data'].shape[0]
    print ("Total dimension:", totdim)
    assert totdim == focks['td_fock_at__im_data'].shape[0]
    assert totdim == denss['td_dens_re_data'].shape[0]
    assert totdim == denss['td_dens_im_data'].shape[0]

    testsetdim = int(0.20*totdim)
    print ("Test set dimension:", testsetdim)
    trainsetdim = totdim - testsetdim
    print ("Train set dimension:", trainsetdim)

    if DEBUG:
        trainsetdim = 200
        testsetdim = 40
        totdim = trainsetdim + testsetdim
        print("DEBUG: Reduced dimensions for testing purposes.")

    X_test = []
    Y_test = []
    for i in range(trainsetdim, totdim):  
        Y_test.append(np.concatenate((focks['td_fock_at_re_data'][i,:,:].flatten(),
                                  focks['td_fock_at__im_data'][i,:,:].flatten())))
        X_test.append(np.concatenate((denss['td_dens_re_data'][i,:,:].flatten(),
                                  denss['td_dens_im_data'][i,:,:].flatten())))
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    X_train = []
    Y_train = []
    for i in range(0, trainsetdim):
        Y_train.append(np.concatenate((focks['td_fock_at_re_data'][i,:,:].flatten(),
                                       focks['td_fock_at__im_data'][i,:,:].flatten())))
        X_train.append(np.concatenate((denss['td_dens_re_data'][i,:,:].flatten(),
                                       denss['td_dens_im_data'][i,:,:].flatten())))
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # check if they are well packed

    # normalize the data using preprocessing and StansdardScaler
    scalerX = preprocessing.StandardScaler()
    X_train = scalerX.fit_transform(X_train)
    scalerY = preprocessing.StandardScaler()
    Y_train = scalerY.fit_transform(Y_train)

    X_test = scalerX.transform(X_test)
    Y_test = scalerY.transform(Y_test)

    # double check the scaled values 

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    modelshapes =  [
        [128, 256, 128, 64],
        [256, 256, 256, 256],
        [256, 256, 256, 256, 256, 256, 256, 256, 256, 256]]
    lossfun = "mse"
    nepochs = 100
    batchsize = 32
    
    for modelshape in modelshapes:
        print("Training model with shape:", modelshape)
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        for layerdim in modelshape:
            model.add(layers.Dense(layerdim, activation='relu'))
        model.add(layers.Dense(Y_train.shape[1], activation='linear'))

        # IF i WANT TO USE THE BEST EPOCHS
        #filepath = 'model_epoch_{epoch}.keras' 
        #save_model_callback = SaveModelEpoch(filepath)

        model.compile(optimizer='adam', loss=lossfun, metrics=[lossfun])

        history = model.fit(X_train, Y_train, epochs=nepochs, \
                            batch_size=batchsize, \
                            validation_split=0.2,\
                            #callbacks=[save_model_callback], \
                            verbose=0)

        #minmse = min(history.history[lossfun])
        #minepoch = np.argmin(history.history[lossfun])
        #epoch_str = '{:04d}'.format(minepoch + 1)  # Format epoch number
        #filename = filepath.format(epoch=epoch_str)
        #model = keras.models.load_model(filename)

        Y_pred = model.predict(X_test)
        Y_pred = scalerY.inverse_transform(Y_pred)
        Y_test = scalerY.inverse_transform(Y_test)

        mape = np.mean(np.abs((Y_test - Y_test) / Y_test)) * 100
        print("Mean Absolute Percentage Error (MAPE):", mape)
        mse = np.mean((Y_test - Y_pred) ** 2)
        print("Mean Squared Error (MSE):", mse)
