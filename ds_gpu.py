import os
from random import randint
from PIL import Image
from multiprocessing import Pool
import numpy as np
import cv2

doc_data = '/gemini/data-1/'
doc_output = '/gemini/output/'

# doc_data = 'data/'
# doc_output = ''

l = os.listdir(doc_data+'train')
l = [i for i in l if os.path.isdir(doc_data+'train/'+i)]
li = []
for p in l:
    li += [doc_data+'train/'+p+'/'+i for i in os.listdir(doc_data+'train/'+p) if i.split('.')[-1]== 'jpg']


X_train = []
Y_train = []
y_train = []
Y_test = []
X_test = []
y_test = []

MAX_VAL = 180
COLOR_table = []
batch_size = 512

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

def _pool_init():
    global tf, batch_size
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for i in gpus:
        # tf.config.experimental.set_virtual_device_configuration(i,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)])
        tf.config.experimental.set_memory_growth(i,True)

def init_proc():
    poo = Pool(processes=8,initializer=_pool_init)
    return poo

## preprocessing training data
def process_tr_data(idx):
    Y_train=[]
    y_train=[]
    for (x, item) in enumerate(li[idx * batch_size:(idx + 1) * batch_size]):
        im = Image.open(item)
        im = im.resize((224,224))
        im=im.convert('L')
        im=im.point(COLOR_table,'L')
        imgarr = np.asarray(im)
        # X_train.append(np.asarray(im.resize((224,224))))
        mask = np.zeros((224,224),dtype=np.uint8)
        r1 = randint(3,16)
        r2 = randint(3,16)
        points1 = random_polygon(40, [randint(r1//2,224-r1//2), randint(r2//2,224-r2//2)], [r1, r2])
        mask = cv2.fillPoly(mask, [points1], (255))
        mask = np.asarray(mask)
        imgarray = np.where(mask>0,-1,imgarr)
        Y_train.append(imgarray)
        # ytr = np.asarray(Image.fromarray(imgarray).resize((224,224)))
        ytr = np.multiply(np.where(imgarray < 0, 1, 0),imgarr)
        y_train.append(ytr)
    with tf.device('/GPU:0'):
        Y_train = tf.constant(Y_train)
        y_train = tf.constant(y_train)
    return Y_train, y_train

def Generator(seq):
    while True:
        for item in seq:
            yield item


def load_train_data(poo: Pool,b_size):
    global tf
    import tensorflow as tf
    l = range(len(li)//batch_size)
    rtn = poo.map(process_tr_data, l)
    poo.close()
    poo.join()
    print('ds: loaded train data!')
    return Generator(rtn),l


# X_train = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
# Y_train = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
# y_train = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)


def process_tst_data():
    global X_test,Y_test,y_test

    li = os.listdir(doc_data+'test')
    li = [i for i in li if i.split('.')[-1]== 'jpg']

    for (x, item) in enumerate(li):
        im = Image.open(doc_data+'test/'+item)
        im = im.resize((224,224))
        im=im.convert('L')
        im=im.point(COLOR_table,'L')
        imgarray = np.asarray(im)
        X_test.append(np.asarray(im.resize((224,224))))
        mask = np.zeros((224,224),dtype=np.uint8)
        r1 = randint(3,63)
        r2 = randint(3,63)
        points1 = random_polygon(40, [randint(r1//2,224-r1//2), randint(r2//2,224-r2//2)], [r1, r2])
        mask = cv2.fillPoly(mask, [points1], (255))
        mask = np.asarray(mask)
        imgarray = np.where(mask>0,-1,imgarray)
        Y_test.append(imgarray)
        yte = np.asarray(Image.fromarray(imgarray).resize((224,224)))
        y_test.append(yte)


    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    y_test = np.array(y_test)

def load_test_data():
    process_tst_data()
    print("ds: loaded test data!")
    return (X_test,Y_test,y_test)