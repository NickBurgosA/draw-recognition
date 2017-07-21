import cv2

import numpy as np
import glob
import os

ancho = 65
alto = 65
resize = (1365,520)
CLASS_N = 3

# local modules
from common import clock, mosaic


def split2d(img, cell_size, resize=None, flatten=True):
    h, w = img.shape[:2]
    print(h,w)
    sx, sy = cell_size
    h, w = img.shape[:2]
    cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_images(fn):
    full_image = cv2.imread(fn, 0)
    images = split2d(full_image, (ancho, alto))
    labels = np.repeat([0,1,2,3], len(images) / 4)
    # ['apple', 'basketball', 'cake', 'cat'] = [0,1,2,3]
    return images,labels


def merge_images(src):
    imagenes = []
    for img in sorted(glob.glob(src)):
        imagen = cv2.imread(img, 0)
        imagen = cv2.resize(imagen, resize)
        imagenes.append(imagen)

    full = np.concatenate(imagenes, axis=0)
    cv2.imwrite('full.png', full)

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C=1, gamma=0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

def evaluate_model(model, images, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Precision: %.2f %%' % ((1 - err) * 100))

    confusion = np.zeros((4, 4), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('Matriz de confusion:')
    print(confusion)

    vis = []
    for img, flag in zip(images, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0

        vis.append(img)
    return mosaic(25, vis)

def calc_hog():
    winSize = (65, 65)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

if __name__ == '__main__':

    merge_images('images/*.png')

    print('Cargando imagenes desde full.png ... ')
    images, labels = load_images('full.png')

    print('Reordenar data ... ')
    rand = np.random.RandomState(3)
    shuffle = rand.permutation(len(images))
    images, labels = images[shuffle], labels[shuffle]

    print('HOG')

    hog = calc_hog()

    print('Calculando el descriptor HOG para cada imagen ... ')
    hog_descriptors = []
    for img in images:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    print('Dividiendo imagenes (90%) entrenamiento y test(10%)... ')
    train_n = int(0.90 * len(hog_descriptors))
    images_train, images_test = np.split(images, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    model = SVM()

    #if not os.path.isfile('draws_svm.dat'):
    print('Entrenando modelo SVM ...')
    model.train(hog_descriptors_train, labels_train)

    print('Guardando el modelo ...')
    model.save('draws_svm.xml')
    #else:
    #model.load('draws_svm.dat')

    print('Evaluando el modelo ... ')

    vis = evaluate_model(model, images_test, hog_descriptors_test, labels_test)
    cv2.imwrite("classification.jpg", vis)
    cv2.imshow("Vis", vis)
    cv2.waitKey(0)




