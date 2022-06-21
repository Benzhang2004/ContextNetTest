from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, merge, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt
import ds
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 227
        self.img_cols = 227
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_gen_shape = (192,192,1)

        optimizer = adam_v2.Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=self.img_shape)
        label = Input(shape=self.img_shape)
        label_gen = Input(shape=self.img_gen_shape)
        img = self.generator([z,label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator([img,label_gen])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z,label, label_gen], validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()
        model.add(Input(shape=self.img_shape))
        model.add(Conv2D(96,(11,11),strides=(4,4),activation='relu',name='conv1'))
        model.add(MaxPool2D((3,3),strides=(2,2),name='maxpool1'))
        model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',name="conv2"))
        model.add(MaxPool2D((3,3),strides=(2,2),name='maxpool2'))
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',name='conv3'))
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',name='conv4'))
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',name='conv5'))
        model.add(MaxPool2D((3,3),strides=(2,2),name='maxpool3'))
        model.add(Flatten())
        model.add(Dense(9216))
        model.add(Reshape((6,6,256)))
        model.add(Conv2DTranspose(128,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2DTranspose(64,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2DTranspose(64,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2DTranspose(32,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2DTranspose(1,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))

        noise = Input(shape=self.img_shape)
        label = Input(shape=self.img_shape)

        model_input = merge.multiply([noise, label])
        img = model(model_input)
        
        return Model([noise,label], img)
        
        

    def build_discriminator(self):

        model = Sequential()
        
        model.add(Input(shape=self.img_gen_shape))
        model.add(Conv2D(32,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(64,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(128,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2D(256,(5,5),strides=(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        img = Input(shape=self.img_gen_shape)
        label = Input(shape=self.img_gen_shape)
        
        tmp = merge.minimum([label,np.zeros((1,)+self.img_gen_shape)])
        tmp = merge.multiply([tmp,img])
        # tmp = merge.multiply([tmp,np.negative(np.ones(self.img_gen_shape))])
        tmp2 = merge.maximum([label,np.zeros((1,)+self.img_gen_shape)])
        comp = merge.add([tmp,tmp2])

        validity = model(comp)
    
        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, Y_train, y_train) = ds.load_data()

        # (XX, 227, 227) -> (XX, 227, 227, 1)
        X_train = np.expand_dims(X_train, axis=3)
        Y_train = np.expand_dims(Y_train, axis=3)
        y_train = np.expand_dims(y_train, axis=3)
        # batch_size = X_train.shape[0]

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            labels = Y_train[idx]
            labels_gen = y_train[idx]

            noise = np.random.normal(0, 1, (batch_size,)+self.img_shape)

            # Generate a batch of new images
            gen_imgs = self.generator.predict([noise,labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs,labels_gen], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs,labels_gen], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size,)+self.img_shape)

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch([noise,labels,labels_gen], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        # r, c = 5, 5
        noise = np.random.normal(0, 1, (1,)+self.img_shape)
        # label = np.asarray([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5])
        # gen_imgs = self.generator.predict([noise,label])

        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("images/%d.png" % epoch)
        # plt.close()
        pass


if __name__ == '__main__':
    gan = GAN()
    gan. train(epochs=30000, batch_size=256, sample_interval=200)