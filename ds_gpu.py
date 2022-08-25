import os
from random import randint
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
import gc
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


# X_train = []
# Y_train = []
# y_train = []
Y_test = []
X_test = []
y_test = []

MAX_VAL = 180
COLOR_table = []
batch_size = 512

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
    pass

## preprocessing training data
def process_tr_data(idx):
    Y_train=[]
    y_train=[]
    for (x, item) in enumerate(li[idx * batch_size:(idx + 1) * batch_size]):
        imd = cv2.imread(item,cv2.IMREAD_GRAYSCALE)
        imd = cv2.resize(imd,(224,224))
        imd = np.where(imd<MAX_VAL,1,0)
        im = np.array(imd,copy=True)
        gc.collect()
        # X_train.append(np.asarray(im.resize((224,224))))
        mask = np.zeros((224,224),dtype=np.uint8)
        r1 = randint(3,16)
        r2 = randint(3,16)
        points1 = random_polygon(40, [randint(r1//2,224-r1//2), randint(r2//2,224-r2//2)], [r1, r2])
        mask = cv2.fillPoly(mask, [points1], (255))
        mask = np.asarray(mask)
        imgarray = np.where(mask>0,-1,im)
        Y_train.append(imgarray)
        # ytr = np.asarray(Image.fromarray(imgarray).resize((224,224)))
        ytr = np.multiply(np.where(imgarray < 0, 1, 0),im)
        y_train.append(ytr)
    Y_train = np.array(Y_train)
    y_train = np.array(y_train)
    a=Y_train.tolist()
    b=y_train.tolist()
    c=[]
    d=[]
    with tf.device('/GPU:0'):
        c = tf.constant(a)
        d = tf.constant(b)
    print('put batch: ',idx)
    gc.collect()
    return c, d

def Generator(seq):
    while True:
        for item in seq:
            yield item


def load_train_data(poo,b_size):
    # global tf
    # import tensorflow as tf
    l = range(len(li)//batch_size)
    # rtn = []
    # for i in l:
    #     rtn.append(process_tr_data(i))
    rtn = poo.map(process_tr_data, l)
    poo.close()
    poo.join()
    print('ds: loaded train data!')
    return Generator(rtn),len(l)


# X_train = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
# Y_train = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
# y_train = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)


def process_tst_data():
    global X_test,Y_test,y_test

    li = os.listdir(doc_data+'test')
    li = [i for i in li if i.split('.')[-1]== 'jpg']

    for (x, item) in enumerate(li):
        im = cv2.imread(doc_data+'test/'+item,cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im,(224,224))
        im = np.where(im<MAX_VAL,1,0)
        X_test.append(np.asarray(im.resize((224,224))))
        mask = np.zeros((224,224),dtype=np.uint8)
        r1 = randint(3,63)
        r2 = randint(3,63)
        points1 = random_polygon(40, [randint(r1//2,224-r1//2), randint(r2//2,224-r2//2)], [r1, r2])
        mask = cv2.fillPoly(mask, [points1], (255))
        mask = np.asarray(mask)
        imgarray = np.where(mask>0,-1,im)
        Y_test.append(imgarray)
        # yte = np.asarray(Image.fromarray(imgarray).resize((224,224)))
        # y_test.append(yte)


    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    y_test = np.array(y_test)

def load_test_data():
    process_tst_data()
    print("ds: loaded test data!")
    return (X_test,Y_test,y_test)