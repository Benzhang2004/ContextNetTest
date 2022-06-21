import os
from random import randint
from PIL import Image
import numpy as np
import cv2
import scipy.misc

li = os.listdir('data/train')
li = [i for i in li if i.split('.')[-1]== 'jpg']


X_train = []
Y_train = []
y_train = []

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



for (x, item) in enumerate(li):
    im = Image.open('data/train/'+item)
    im = im.resize((227,227))
    im=im.convert('L')
    im=im.point(COLOR_table,'L')
    imgarray = np.asarray(im)
    X_train.append(np.asarray(im.resize((192,192))))
    mask = np.zeros((227,227),dtype=np.uint8)
    r1 = randint(10,114)
    r2 = randint(10,114)
    points1 = random_polygon(40, [randint(r1//2,227-r1//2), randint(r2//2,227-r2//2)], [r1, r2])
    mask = cv2.fillPoly(mask, [points1], (255))
    mask = np.asarray(mask)
    imgarray = np.where(mask>0,-1,imgarray)
    Y_train.append(imgarray)
    ytr = np.asarray(Image.fromarray(imgarray).resize((192,192)))
    y_train.append(ytr)
    

X_train = np.array(X_train)
Y_train = np.array(Y_train)
y_train = np.array(y_train)

def load_data():
    return (X_train,Y_train,y_train)


# def draw_polygon(image_size, points, color):
#     image = np.zeros(image_size, dtype=np.uint8)
#     if type(points) is np.ndarray and points.ndim == 2:
#         image = cv2.fillPoly(image, [points], color)
#     else:
#         image = cv2.fillPoly(image, points, color)
#     return image



# points1 = random_polygon(10, [80, 80], [70, 70])
# points2 = random_polygon(10, [80, 180], [20, 50])
# points3 = random_polygon(3, [180, 80], [20, 50])
# points4 = random_polygon(5, [180, 180], [20, 50])

# pts = [points1, points2, points3, points4]

# # image1 = draw_polygon((256, 256, 3), points1, (255, 255, 255))
# image2 = draw_polygon((227, 227, 1), pts, (255))
# # cv2.imshow('a', image1)
# cv2.imshow('b', image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

