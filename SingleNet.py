from unicodedata import name
from keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Activation, ReLU, MaxPool2D, Dropout
from keras.models import Sequential, Model
from keras.optimizers import adam_v2
import tensorflow as tf
import keras.layers.merge as merge
import ds_single as ds
import numpy as np
import matplotlib.pyplot as plt
import os

class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_gen_shape = (64,64,1)

        optimizer1 = adam_v2.Adam(0.00002, 0.5)
        optimizer2 = adam_v2.Adam(0.1, 0.5)
        optimizer3 = adam_v2.Adam(0.01, 0.5)

        # Create Dirs
        if(not os.path.exists('models')):
            os.mkdir('models')
        if(not os.path.exists('images')):
            os.mkdir('images')

        self.epo = 0
        if(os.path.exists('models/epoch')):
            with open('models/epoch','r') as f:
                self.epo = int(f.read())

        # Build the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer1,
            metrics=['accuracy'])

        # Load the discriminator weights
        if(os.path.exists('models/dis.h5')):
            self.discriminator.load_weights('models/dis.h5')

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',
            optimizer=optimizer2)

        # Load the generator weights
        if(os.path.exists('models/gen.h5')):
            self.generator.load_weights('models/gen.h5')

        # The generator takes noise as input and generates imgs
        label = Input(shape=self.img_shape)
        img = self.generator(label)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(label, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer3)

        # Load the dataset
        (self.X_train,self.Y_train,self.y_train),(self.X_test, self.Y_test, self.y_test) = ds.load_data()


    def build_generator(self):

        model = Sequential()

        model.add(Conv2D(96,11,4,padding='valid')) # conv1
        model.add(MaxPool2D(3,2,padding='valid')) # maxpool1
        model.add(Conv2D(256,5,1,padding='same')) # conv2
        # model.add(MaxPool2D(3,2,padding='valid')) # maxpool2
        model.add(Conv2D(384,3,1,padding='same')) # conv3
        model.add(Conv2D(384,3,1,padding='same')) # conv4
        model.add(Conv2D(256,3,1,padding='same')) # conv5
        # model.add(MaxPool2D(3,2,padding='valid')) # maxpool5


        # model.add(Conv2D(64,(4,4),(2,2),padding='same'))
        # model.add(ReLU()) # 32
        # model.add(Conv2D(128,(4,4),(2,2),padding='same'))
        # model.add(ReLU()) # 16
        # model.add(Conv2D(256,(4,4),(2,2),padding='same'))
        # model.add(ReLU()) # 8
        # model.add(Conv2D(512,(4,4),(2,2),padding='same'))
        # model.add(ReLU()) # 4
        # model.add(Conv2D(1024,(4,4),(1,1),padding='same'))
        # model.add(ReLU()) # 4
        model.add(Flatten())
        model.add(Dense(16384))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Reshape((4,4,1024)))
        model.add(Conv2DTranspose(512,(4,4),(2,2),padding='same'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(ReLU()) # 8
        model.add(Conv2DTranspose(256,(4,4),(2,2),padding='same'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(ReLU()) # 16
        model.add(Conv2DTranspose(128,(4,4),(2,2),padding='same'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(ReLU()) # 32
        model.add(Conv2DTranspose(1,(4,4),(2,2),padding='same'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(ReLU()) # 64
        model.add(Activation('sigmoid'))

        label = Input(shape=self.img_shape)

        img = model(label)
        
        return Model(label, img)
        
        

    def build_discriminator(self):

        model = Sequential()
        
        model.add(Conv2D(256,(4,4),(2,2),padding='same'))
        model.add(ReLU())
        model.add(Conv2D(512,(4,4),(2,2),padding='same'))
        model.add(ReLU())
        model.add(Flatten())
        model.add(Dense(16384))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_gen_shape)

        validity = model(img)
    
        return Model(img, validity)



    def train(self, epochs, batch_size=128, sample_interval=50):

        # (XX, 227, 227) -> (XX, 227, 227, 1)
        Yytrain = ds.SingleNetTrDS(batch_size)

        epoch = self.epo
        for i in range(int(epochs/sample_interval)):
            self.generator.fit(Yytrain, epochs=sample_interval)
            epoch+=sample_interval
            self.cur_iter = epoch
            self.sample_images(epoch)
            self.save_models()
            
            

    def sample_images(self, epoch):
        idx = np.random.randint(0, self.X_test.shape[0], 1)
        label = self.Y_test[idx]
        gen_img = self.generator.predict(label)

        # Rescale images 0 - 1
        gen_img = np.minimum(np.maximum(gen_img,0),1)

        # Shaped images
        shaped_img = np.add(np.multiply(np.where(label[0,:,:] < 0, 1, 0),gen_img[0,:,:,0]),np.maximum(label[0,:,:],0))

        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(self.X_test[idx][0,:,:], cmap='gray')
        axs[1].imshow(label[0,:,:], cmap='gray')
        axs[2].imshow(gen_img[0,:,:,0], cmap='gray')
        axs[3].imshow(shaped_img, cmap='gray')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        axs[3].axis('off')
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_models(self):
        self.generator.save_weights('models/gen.h5')
        self.discriminator.trainable = True
        self.discriminator.save_weights('models/dis.h5')
        with open('models/epoch','w') as f:
            f.write(str(self.cur_iter))
        self.discriminator.trainable = False


if __name__ == '__main__':
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=32510)])

    gan = GAN()
    gan. train(epochs=100000, batch_size=512, sample_interval=10)