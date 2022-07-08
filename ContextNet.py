from keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU
from keras.layers import BatchNormalization
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
        self.img_gen_shape = (64,64,1)

        optimizer = adam_v2.Adam(0.0002, 0.5)

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
            optimizer=optimizer,
            metrics=['accuracy'])

        # Load the discriminator weights
        if(os.path.exists('models/dis.h5')):
            self.discriminator.load_weights('models/dis.h5')

        # Build the generator
        self.generator = self.build_generator()

        # Load the generator weights
        if(os.path.exists('models/gen.h5')):
            self.generator.load_weights('models/gen.h5')

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

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(8192))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(8192))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_gen_shape), activation='tanh'))
        model.add(Reshape(self.img_gen_shape))

        noise = Input(shape=self.img_shape)
        label = Input(shape=self.img_shape)

        model_input = merge.multiply([noise, label])
        img = model(model_input)
        
        return Model([noise,label], img)
        
        

    def build_discriminator(self):

        model = Sequential()
        
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(2048))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_gen_shape)
        label = Input(shape=self.img_gen_shape)
        
        tmp = merge.minimum([label,np.zeros((1,)+self.img_gen_shape)])
        tmp = merge.multiply([tmp,img])
        tmp = merge.multiply([tmp,np.multiply(np.ones((1,)+self.img_gen_shape),-1)])
        tmp2 = merge.maximum([label,np.zeros((1,)+self.img_gen_shape)])
        comp = merge.add([tmp,tmp2])

        validity = model(comp)
    
        return Model([img, label], validity)



    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (_,_,_),(self.X_test, self.Y_test, self.y_test) = ds.load_data()

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
            d_loss_real = self.discriminator.fit([imgs,labels_gen], valid, epochs=2, verbose=0)
            d_loss_fake = self.discriminator.fit([gen_imgs,labels_gen], fake, epochs=2, verbose=0)
            d_loss = 0.5 * (d_loss_real.history['loss'][-1] + d_loss_fake.history['loss'][-1])
            d_loss_acc = 0.5 * (d_loss_real.history['accuracy'][-1] + d_loss_fake.history['accuracy'][-1])

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size,)+self.img_shape)

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.fit([noise,labels,labels_gen], valid, epochs=10, verbose=0)
            
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss, 100*d_loss_acc, g_loss.history['loss'][-1]))

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
        gen_img = np.maximum(gen_img,0)

        # Shaped images
        tmp = np.minimum(label_gen[0,:,:],0)
        tmp = np.multiply(tmp,gen_img[0,:,:,0])
        tmp = np.multiply(tmp,-1)
        tmp2 = np.maximum(label_gen[0,:,:],0)
        shaped_img = np.add(tmp,tmp2)

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


if __name__ == '__main__':
    gan = GAN()
    gan. train(epochs=100000, batch_size=512, sample_interval=50)