from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
import os
import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Flatten,Dropout


import numpy as np
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
import os


ROOT_PATH = "/home/wlyu/Downloads/newdir/traffic"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")
save_path='traffic.h5'
load_path='traffic.h5'
def load_data(data_dir):
    """Loads a data set and returns two lists:
    
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
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
    return images, labels
def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()
def display_label_images(images, label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()


#load and transform data
images, labels = load_data(train_data_dir) 
test_images, test_labels_ = load_data(test_data_dir)
images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]
test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]   
test_images32_a=np.array(test_images32)
test_labels=np.array(test_labels_)
test_labels = keras.utils.to_categorical(test_labels, num_classes=62)



sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
labels_a = np.array(labels)
labels_a = keras.utils.to_categorical(labels_a, num_classes=62)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)



def build_model(model):
	layers=[Dense(64, activation='relu', input_shape=images_a.shape[1:]),
			Dropout(0.5),
			Flatten(),
			Dense(62)]					
	for layer in layers:
	        model.add(layer)

	model.add(Activation('softmax'))        
	model.compile(optimizer='rmsprop',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

try:
	model=load_model(load_path)
except IOError,OSError:
	# print(OSError)
	print("New model will be built!\n")
	model=Sequential()
	build_model(model)
	model.fit(images_a, labels_a, epochs=10, batch_size=32)
	model.save(save_path)


score = model.evaluate(test_images32_a, test_labels, batch_size=128)




images_ph=tf.placeholder(tf.float32, [None, 32, 32, 3])

sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]
predicted_labels=model(images_ph)



predicted=predicted_labels.eval(feed_dict={images_ph: sample_images})
predicted=list(np.argmax(predicted,axis=1))
print(sample_labels)
print(predicted)

plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])
plt.show()



wrap = KerasModelWrapper(model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.03,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate(images_ph, **fgsm_params)
# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)
preds_adv = model(adv_x)

ADV=adv_x.eval(feed_dict={images_ph:sample_images})
PRED_ADV=preds_adv.eval(feed_dict={images_ph:sample_images})
PRED_ADV=np.argmax(PRED_ADV,axis=1)

plt.figure(figsize=(10, 10))
for i in range(len(ADV)):
    truth = sample_labels[i]
    prediction = PRED_ADV[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(ADV[i])
plt.show()
# batch_size=128
# Evaluate the accuracy of the MNIST model on adversarial examples
# eval_par = {'batch_size': batch_size}
# acc = model_eval(sess, images_ph, y, preds_adv, X_test, Y_test, args=eval_par)
# print('Test accuracy on adversarial examples: %0.4f\n' % acc)
predicted_adv = sess.run([preds_adv], 
                        feed_dict={images_ph: test_images32_a})[0]
predicted_adv=np.argmax(predicted_adv,axis=1)
# print(predicted_adv.)
# print(predicted_adv)
# print(test_labels_)
# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels_, list(predicted_adv))])
accuracy_adv = match_count / len(test_labels_)



print(accuracy_adv)










sess.close()
