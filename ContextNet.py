from keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Activation, ReLU
from keras.models import Sequential, Model
from keras.optimizers import adam_v2
import tensorflow as tf
import keras.layers.merge as merge
import ds
import numpy as np
import matplotlib.pyplot as plt
import os

class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_gen_shape = (16,16,1)

        optimizer1 = adam_v2.Adam(0.00002, 0.5)
        optimizer2 = adam_v2.Adam(0.04, 0.5)
        optimizer3 = adam_v2.Adam(0.02, 0.5)

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
        z = Input(shape=self.img_shape)
        label = Input(shape=self.img_shape)
        img = self.generator([z,label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z,label], validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer3)

        # Load the dataset
        (_,_,_),(self.X_test, self.Y_test, self.y_test) = ds.load_data()


    def build_generator(self):

        model = Sequential()

        model.add(Conv2D(64,(4,4),(2,2),padding='same'))
        model.add(ReLU())
        model.add(Conv2D(128,(4,4),(2,2),padding='same'))
        model.add(ReLU())
        model.add(Conv2D(256,(4,4),(2,2),padding='same'))
        model.add(ReLU())
        model.add(Conv2D(512,(4,4),(2,2),padding='same'))
        model.add(ReLU())
        model.add(Flatten())
        model.add(Dense(16384))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Reshape((4,4,1024)))
        model.add(Conv2DTranspose(512,(4,4),(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv2DTranspose(1,(4,4),(2,2),padding='same'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Activation('sigmoid'))

        noise = Input(shape=self.img_shape)
        label = Input(shape=self.img_shape)

        model_input = merge.multiply([noise, label])

        img = model(model_input)
        
        return Model([noise,label], img)
        
        

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
        ds.epo = self.epo
        ds.batch_size = batch_size
        ds.epochs = epochs
        ds.init_data()

        # Load TF dataset
        tfXtrain = tf.data.Dataset.from_generator(ds._gen_Xtrain,output_signature=(tf.TensorSpec(shape=self.img_gen_shape,dtype=tf.float32)),args=())
        tfYtrain = tf.data.Dataset.from_generator(ds._gen_Ytrain,output_signature=(tf.TensorSpec(shape=self.img_shape,dtype=tf.float32)),args=())
        tfytrain = tf.data.Dataset.from_generator(ds._gen_ytrain,output_signature=(tf.TensorSpec(shape=self.img_gen_shape,dtype=tf.float32)),args=())
        dataset = tf.data.Dataset.zip((tfXtrain,tfYtrain,tfytrain)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        epoch = self.epo
        for item in dataset:
            
            # Interrupt training
            # if(keyboard.is_pressed('q')):
                # break

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            imgs = item[0]
            labels = item[1]
            labels_gen = item[2]

            noise = np.random.normal(0, 1, (batch_size,)+self.img_shape)

            # Generate a batch of new images
            gen_imgs = self.generator.predict([noise,labels])

            # Train the discriminator
            d_loss_real = self.discriminator.fit([imgs], valid, epochs=4, verbose=0)
            d_loss_fake = self.discriminator.fit([gen_imgs], fake, epochs=4, verbose=0)
            d_loss = 0.5 * (d_loss_real.history['loss'][-1] + d_loss_fake.history['loss'][-1])
            d_loss_acc = 0.5 * (d_loss_real.history['accuracy'][-1] + d_loss_fake.history['accuracy'][-1])

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size,)+self.img_shape)

            # Train the generator (to have the discriminator label samples as valid)
            c_loss = self.generator.fit([noise,labels], imgs, epochs=9, verbose=0)
            g_loss = self.combined.fit([noise,labels], valid, epochs=1, verbose=0)
            
            # Plot the progress
            print ("[C loss: ", c_loss.history['loss'][-1],end='] ')
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100*d_loss_acc, g_loss.history['loss'][-1]))
            # print ("%d [D loss: %f, acc.: %.2f%%]" % (epoch, d_loss, 100*d_loss_acc))


            self.cur_iter = epoch

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        
        # Save Models
                self.save_models()
            

            epoch += 1
            

    def sample_images(self, epoch):
        idx = np.random.randint(0, self.X_test.shape[0], 1)
        noise = np.random.normal(0, 1, (1,)+self.img_shape)
        label = self.Y_test[idx]
        gen_img = self.generator.predict([noise,label])
        label_gen = self.y_test[idx]

        # Rescale images 0 - 1
        gen_img = np.minimum(np.maximum(gen_img,0),1)

        # Shaped images
        tmp = np.pad(gen_img[0,:,:,0],((24,24),(24,24)))
        shaped_img = np.add(np.maximum(label[0,:,:],0),tmp)

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
    tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11264)])

    gan = GAN()
    gan. train(epochs=100000, batch_size=512, sample_interval=50)