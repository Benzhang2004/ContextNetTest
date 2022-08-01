import ds
import tensorflow as tf
import numpy as np

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