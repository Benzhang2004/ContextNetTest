import os
from random import randint
from PIL import Image
import numpy as np
import cv2

l = os.listdir('data/train')
l = [i for i in l if os.path.isdir('data/train/'+i)]
li = []
for p in l:
    li += ['data/train/'+p+'/'+i for i in os.listdir('data/train/'+p) if i.split('.')[-1]== 'jpg']


X_train = []
Y_train = []
y_train = []
Y_test = []
X_test = []
y_test = []

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


## preprocessing training data
for (x, item) in enumerate(li):
    im = Image.open(item)
    im = im.resize((64,64))
    im=im.convert('L')
    im=im.point(COLOR_table,'L')
    imgarray = np.asarray(im)
    # X_train.append(np.asarray(im.resize((64,64))))
    mask = np.zeros((64,64),dtype=np.uint8)
    r1 = randint(3,16)
    r2 = randint(3,16)
    points1 = random_polygon(40, [randint(r1//2,64-r1//2), randint(r2//2,64-r2//2)], [r1, r2])
    mask = cv2.fillPoly(mask, [np.array([[24,24],[40,24],[40,40],[24,40]])], (255))
    mask = np.asarray(mask)
    X_train.append(imgarray[24:41,24:41])
    # Image.fromarray(imgarray[24:41,24:41]*255).show()
    imgarray = np.where(mask>0,-1,imgarray)
    Y_train.append(imgarray)
    ytr = np.asarray(Image.fromarray(imgarray).resize((8,8)))
    y_train.append(ytr)


li = os.listdir('data/test')
li = [i for i in li if i.split('.')[-1]== 'jpg']

for (x, item) in enumerate(li):
    im = Image.open('data/test/'+item)
    im = im.resize((64,64))
    im=im.convert('L')
    im=im.point(COLOR_table,'L')
    imgarray = np.asarray(im)
    # X_test.append(np.asarray(im.resize((64,64))))
    mask = np.zeros((64,64),dtype=np.uint8)
    r1 = randint(3,16)
    r2 = randint(3,16)
    points1 = random_polygon(40, [randint(r1//2,64-r1//2), randint(r2//2,64-r2//2)], [r1, r2])
    mask = cv2.fillPoly(mask, [np.array([[24,24],[40,24],[40,40],[24,40]])], (255))
    mask = np.asarray(mask)
    X_test.append(imgarray[24:41,24:41])
    # Image.fromarray(imgarray[24:41,24:41]).show()
    imgarray = np.where(mask>0,-1,imgarray)
    Y_test.append(imgarray)
    yte = np.asarray(Image.fromarray(imgarray).resize((8,8)))
    y_test.append(yte)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
y_test = np.array(y_test)

def load_data():
    return (X_train,Y_train,y_train),(X_test,Y_test,y_test)


def remask(idx):
    imgarray = X_train[idx,:,:,0]
    mask = np.zeros((64,64),dtype=np.uint8)
    r1 = randint(3,16)
    r2 = randint(3,16)
    points1 = random_polygon(40, [randint(r1//2,64-r1//2), randint(r2//2,64-r2//2)], [r1, r2])
    mask = cv2.fillPoly(mask, [np.array([[24,24],[40,24],[40,40],[24,40]])], (255))
    mask = np.asarray(mask)
    imgarray = np.where(mask>0,-1,imgarray)
    global Y_train, y_train
    Y_train[idx] = np.expand_dims(imgarray, axis=2)
    # ytr = np.asarray(Image.fromarray(imgarray).resize((64,64)))
    ytr = imgarray
    y_train[idx] = np.expand_dims(ytr, axis=2)


# Train datasets generators

epo_cur = 0

epo = 0
idxs = []
genX_cur = 0
genY_cur = 0
geny_cur = 0
batch_size = 0
epochs = 0

def init_data():
    global X_train,Y_train,y_train,idxs
    X_train = np.expand_dims(X_train, axis=3)
    Y_train = np.expand_dims(Y_train, axis=3)
    y_train = np.expand_dims(y_train, axis=3)
    idxs += list(np.random.randint(0, X_train.shape[0], batch_size*10))

def _gen_Xtrain():
    global genX_cur
    for i in range(epo, epochs):
        for j in range(batch_size):
            Xx = X_train[idxs[genX_cur]]
            if(genX_cur<min(genY_cur,geny_cur)):
                pass
                # remask(idxs[genX_cur])
            genX_cur+=1
            yield Xx
            gc()

def _gen_Ytrain():
    global genY_cur
    for i in range(epo, epochs):
        for j in range(batch_size):
            Yx = Y_train[idxs[genY_cur]]
            if(genY_cur<min(genX_cur,geny_cur)):
                pass
                # remask(idxs[genY_cur])
            genY_cur+=1
            yield Yx
            gc()

def _gen_ytrain():
    global geny_cur
    for i in range(epo, epochs):
        for j in range(batch_size):
            y = y_train[idxs[geny_cur]]
            if(geny_cur<min(genY_cur,genX_cur)):
                pass
                # remask(idxs[geny_cur])
            geny_cur+=1
            yield y
            gc()

def gc():
    global genX_cur,genY_cur,geny_cur, idxs
    a = min(genX_cur,genY_cur,geny_cur)
    idxs = idxs[a:]
    genX_cur -= a
    genY_cur -= a
    geny_cur -= a
    if(max(genX_cur,genY_cur,geny_cur)>len(idxs)-5):
        idxs += list(np.random.randint(0, X_train.shape[0], batch_size*10))