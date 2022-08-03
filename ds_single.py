from random import randint
import ds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train,Y_train,y_train),(X_test,Y_test,y_test) = ds.load_data()
y_train = np.multiply(np.where(Y_train < 0, 1, 0),X_train)
ds.init_data()

def load_data():
    return ds.load_data()

class SingleNetTrDS(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return int((ds.X_train.shape[0])/self.batch_size)

    def __getitem__(self, idx):
        batch_Y = Y_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = y_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            ds.remask(i)
            Y_train[i] = ds.Y_train[i,:,:,0]
            y_train[i] = np.multiply(np.where(Y_train[i] < 0, 1, 0),X_train[i])
        return batch_Y, batch_y

class SingleNetPRDS(tf.keras.utils.Sequence):
    def __init__(self, batch_size, maxlen=500):
        self.batch_size = batch_size
        self.seq = randint(0,X_train.shape[0])
        self.MAXLEN = maxlen

    def __len__(self):
        return self.MAXLEN

    def __getitem__(self, idx):

        batch_Y = [Y_train[self.seq]]
        batch_y = [y_train[self.seq]]
        for i in range(self.batch_size-1):
            ds.remask(self.seq)
            batch_Y.append(ds.Y_train[self.seq,:,:,0])
            batch_y.append(np.multiply(np.where(ds.Y_train[self.seq,:,:,0] < 0, 1, 0),X_train[self.seq]))
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(ds.Y_train[self.seq,:,:,0], cmap='gray')
            axs[1].imshow(np.multiply(np.where(ds.Y_train[self.seq,:,:,0] < 0, 1, 0),X_train[self.seq]), cmap='gray')
            axs[0].axis('off')
            axs[1].axis('off')
            fig.savefig("images/idx"+str(idx)+".png")

        ds.remask(self.seq)
        Y_train[self.seq] = ds.Y_train[self.seq,:,:,0]
        y_train[self.seq] = np.multiply(np.where(Y_train[self.seq] < 0, 1, 0),X_train[self.seq])
        return np.array(batch_Y), np.array(batch_y)

    def on_epoch_end(self):
        self.seq = randint(0,X_train.shape[0])