import os
from random import randint
import ds
import tensorflow as tf
import numpy as np
from PIL import Image 
import cv2

(_,_,_),(X_test,Y_test,y_test) = ds.load_data()

l = os.listdir('/gemini/data-1/train')
l = [i for i in l if os.path.isdir('/gemini/data-1/train/'+i)]
li = []
for p in l:
    li += ['/gemini/data-1/train/'+p+'/'+i for i in os.listdir('/gemini/data-1/train/'+p) if i.split('.')[-1]== 'jpg']

MAX_VAL = 180
COLOR_table = []
for i in range(256):
    if(i<MAX_VAL):
        COLOR_table.append(1)
    else:
        COLOR_table.append(0)

def uniform_random(left, right, size=None):
    rand_nums = (right - left) * np.random.random(size) + left
    return rand_nums

def random_polygon(edge_num, center, radius_range):
    angles = uniform_random(0, 2 * np.pi, edge_num)
    angles = np.sort(angles)
    random_radius = uniform_random(radius_range[0], radius_range[1], edge_num)
    x = np.cos(angles) * random_radius
    y = np.sin(angles) * random_radius
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    points = np.concatenate([x, y], axis=1)
    points += np.array(center)
    points = np.round(points).astype(np.int32)
    return points

def load_data():
    return ds.load_data()

class SingleNetTrDS(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return int((len(li))/self.batch_size)

    def __getitem__(self, idx):

        batch_Y = []
        batch_y = []

        for (x, item) in enumerate(li[idx * self.batch_size:(idx + 1) * self.batch_size]):
            im = Image.open(item)
            im = im.resize((224,224))
            im=im.convert('L')
            im=im.point(COLOR_table,'L')
            imgarray = np.asarray(im)
            mask = np.zeros((224,224),dtype=np.uint8)
            r1 = randint(3,63)
            r2 = randint(3,63)
            points1 = random_polygon(40, [randint(r1//2,224-r1//2), randint(r2//2,224-r2//2)], [r1, r2])
            mask = cv2.fillPoly(mask, [points1], (255))
            mask = np.asarray(mask)
            imgarray_s = np.where(mask>0,-1,imgarray)
            batch_Y.append(imgarray_s)
            batch_y.append(np.multiply(np.where(np.array(imgarray_s) < 0, 1, 0),imgarray))

        print('load_data: ',idx)

        batch_Y = np.array(batch_Y)
        batch_y = np.array(batch_y)
        return batch_Y, batch_y

# class SingleNetTrDS_Noshade(tf.keras.utils.Sequence):
#     def __init__(self, batch_size):
#         self.batch_size = batch_size

#     def __len__(self):
#         return int((ds.X_train.shape[0])/self.batch_size)

#     def __getitem__(self, idx):
#         batch_Y = Y_train[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_X = X_train[idx * self.batch_size:(idx + 1) * self.batch_size]
#         for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
#             ds.remask(i)
#             Y_train[i] = ds.Y_train[i,:,:,0]
#         return batch_Y, batch_X

# class SingleNetPRDS(tf.keras.utils.Sequence):
#     def __init__(self, batch_size, maxlen=500):
#         self.batch_size = batch_size
#         self.seq = randint(0,X_train.shape[0])
#         self.MAXLEN = maxlen

#     def __len__(self):
#         return self.MAXLEN

#     def __getitem__(self, idx):

#         batch_Y = [Y_train[self.seq]]
#         batch_y = [y_train[self.seq]]
#         for i in range(self.batch_size-1):
#             ds.remask(self.seq)
#             batch_Y.append(ds.Y_train[self.seq,:,:,0])
#             batch_y.append(np.multiply(np.where(ds.Y_train[self.seq,:,:,0] < 0, 1, 0),X_train[self.seq]))
#             fig, axs = plt.subplots(1, 2)
#             axs[0].imshow(ds.Y_train[self.seq,:,:,0], cmap='gray')
#             axs[1].imshow(np.multiply(np.where(ds.Y_train[self.seq,:,:,0] < 0, 1, 0),X_train[self.seq]), cmap='gray')
#             axs[0].axis('off')
#             axs[1].axis('off')
#             fig.savefig("images/idx"+str(idx)+".png")

#         ds.remask(self.seq)
#         Y_train[self.seq] = ds.Y_train[self.seq,:,:,0]
#         y_train[self.seq] = np.multiply(np.where(Y_train[self.seq] < 0, 1, 0),X_train[self.seq])
#         return np.array(batch_Y), np.array(batch_y)

#     def on_epoch_end(self):
#         self.seq = randint(0,X_train.shape[0])