from keras.models import load_model
import keras
import numpy as np
import os

import skimage.data,skimage.transform
def load_traffic_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    images = [skimage.transform.resize(image, (32, 32))
                 for image in images]   
    images = np.asarray(images)
    labels = np.asarray(labels)
    labels = keras.utils.to_categorical(labels, num_classes=62)
    return images, labels

def classify(X,dataset='mnist'):
    print dataset
    assert dataset in ['mnist','roadsign']
    if dataset =='mnist':
        X = X.reshape((-1,28,28,1))
        model = load_model("model/model_mnist.h5")
    else:
        X = X.reshape((-1,32,32,3))
        model = load_model("model/model_traffic.h5")
    Y = model.predict_classes(X)
    return Y
if __name__=="__main__":
    X,Y = load_traffic_data("data/traffic")
    print X.shape,Y.shape
    X1 = X[3]
    Y1 = Y[3]
    print X1.shape,X1
    print Y1.shape,Y1
    print classify(X1,'roadsign')
