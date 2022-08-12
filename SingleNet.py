from unicodedata import name
from keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Activation, ReLU, MaxPool2D, Dropout
from keras.models import Sequential, Model
from keras.optimizers import adam_v2
import tensorflow as tf
import keras.layers.merge as merge
import ds_gpu as ds
import numpy as np
import matplotlib.pyplot as plt
import os

class GAN():
    def __init__(self, data, output):
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_gen_shape = (224,224,1)
        self.data = data
        self.output = output

        optimizer = adam_v2.Adam(0.1, 0.5)

        # Create Dirs
        if(not os.path.exists(self.output+'models')):
            os.mkdir(self.output+'models')
        if(not os.path.exists(self.output+'images')):
            os.mkdir(self.output+'images')

        self.epo = 0
        if(os.path.exists(self.output+'models/epoch')):
            with open(self.output+'models/epoch','r') as f:
                self.epo = int(f.read())

        # strategy = tf.distribute.MirroredStrategy()
        # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # with strategy.scope():

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',
            optimizer=optimizer)

        # Load the generator weights
        if(os.path.exists(self.output+'models/gen.h5')):
            self.generator.load_weights(self.output+'models/gen.h5')


        # Load the dataset
        (self.X_test, self.Y_test, self.y_test) = ds.load_test_data()


    def build_generator(self):

        model = Sequential()

        model.add(Conv2D(96,11,4,padding='valid')) # conv1
        model.add(MaxPool2D(3,2,padding='valid')) # maxpool1
        model.add(Conv2D(256,5,1,padding='same')) # conv2
        model.add(MaxPool2D(3,2,padding='valid')) # maxpool2
        model.add(Conv2D(384,3,1,padding='same')) # conv3
        model.add(Conv2D(384,3,1,padding='same')) # conv4
        model.add(Conv2D(256,3,1,padding='same')) # conv5
        model.add(MaxPool2D(3,2,padding='valid')) # maxpool5


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
        # model.add(Flatten())
        # model.add(Dense(16384))
        # model.add(BatchNormalization())
        # model.add(ReLU())
        # model.add(Reshape((4,4,1024)))
        # model.add(Conv2DTranspose(512,(4,4),(4,4),padding='same'))
        # model.add(BatchNormalization())
        # model.add(ReLU()) # 16
        # model.add(Conv2DTranspose(1,(4,4),(4,4),padding='same'))
        # model.add(BatchNormalization())
        # model.add(ReLU()) # 64
        # model.add(Activation('sigmoid'))


        # model.add(Conv2D(64,(4,4),(2,2),padding='same'))
        # model.add(Conv2D(128,(4,4),(2,2),padding='same'))
        # model.add(Conv2D(256,(4,4),(2,2),padding='same'))
        # model.add(Conv2D(512,(4,4),(2,2),padding='same'))
        # model.add(Conv2D(512,(4,4),(1,1),padding='same'))
        model.add(Flatten())
        model.add(Dense(29400))
        model.add(BatchNormalization())
        model.add(Reshape((7,7,600)))
        model.add(Conv2DTranspose(512,(4,4),(2,2),padding='same'))
        model.add(BatchNormalization()) # 14
        model.add(Conv2DTranspose(256,(4,4),(2,2),padding='same'))
        model.add(BatchNormalization()) # 28
        model.add(Conv2DTranspose(128,(4,4),(2,2),padding='same'))
        model.add(BatchNormalization()) # 56
        model.add(Conv2DTranspose(64,(4,4),(2,2),padding='same'))
        model.add(BatchNormalization()) # 112
        model.add(Conv2DTranspose(1,(4,4),(2,2),padding='same'))
        model.add(BatchNormalization()) # 224
        model.add(Activation('sigmoid'))

        label = Input(shape=self.img_shape)

        img = model(label)
        # img = model2(merge.multiply([res,Dense(9216)(Flatten()(label))]))
        
        return Model(label, img)


    def train(self, epochs, batch_size=128, sample_interval=50):

        # (XX, 227, 227) -> (XX, 227, 227, 1)
        Yytrain, num = ds.load_train_data(batch_size)

        epoch = self.epo

        # Auto Train
        # for i in range(int(epochs/sample_interval)):
        #     self.generator.fit(Yytrain, epochs=sample_interval, max_queue_size=100, workers=20, use_multiprocessing=True)
        #     epoch+=sample_interval
        #     self.cur_iter = epoch
        #     self.sample_images(epoch)
        #     self.save_models()
        
        # Customized Train
        for i in range(int(epochs/sample_interval)):
            for h in range(1,sample_interval):
                print("Epoch: "+str(h))
                print('',end='')
                for j in range(num):
                    Y,y = next(Yytrain)
                    loss = self.generator.train_on_batch(Y,y)
                    print('\r')
                    print(str(j+1)+'/'+str(len(Yytrain))+'\t'+"Loss: "+str(loss),end='')
                print('')
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
        fig.savefig(self.output+"images/%d.png" % epoch)
        plt.close()

    def save_models(self):
        self.generator.save_weights(self.output+'models/gen.h5')
        with open(self.output+'models/epoch','w') as f:
            f.write(str(self.cur_iter))


if __name__ == '__main__':
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for i in gpus:
        # tf.config.experimental.set_virtual_device_configuration(i,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)])
        tf.config.experimental.set_memory_growth(i,True)

    # gan = GAN('/gemini/data-1/','/gemini/output/')
    gan = GAN('data/','')
    gan. train(epochs=100000, batch_size=512, sample_interval=10)