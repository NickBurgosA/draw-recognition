import cv2

import numpy as np
import glob


ancho = 65
alto = 67
resize = (1365,536)
CLASS_N = 3

# local modules
from common import clock, mosaic


def split2d(img, cell_size, resize=None, flatten=True):
    h, w = img.shape[:2]
    print(h,w)
    sx, sy = cell_size
    img = cv2.resize(img, resize)
    h, w = img.shape[:2]
    cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_images(fn):
    full_image = cv2.imread(fn, 0)
    images = split2d(full_image, (ancho, alto), resize)
    #labels = np.repeat(np.arange(CLASS_N), len(digits) / CLASS_N)
    return images

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969

    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C=12.5, gamma=0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err) * 100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0

        vis.append(img)
    return mosaic(25, vis)


def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ * SZ) / 255.0





full_image = cv2.imread('images/basketball.2.png', 0)
imagenes= load_images('images/basketball.2.png')
cv2.imshow('title',imagenes[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(np.repeat('cat', len(imagenes)))

