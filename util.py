from PIL import Image
import numpy as np
import keras.backend as K
from keras.models import load_model
from attacks import basic_iterative_method    

def png2cvs(filename,dataset='mnist'):
    assert dataset in ["mnist","roadsign"]
    if dataset == 'mnist':
        im = Image.open(filename)
        weight,height = im.size
        tv = list(im.getdata())
        tva = np.asarray([(x)*1.0/255.0 for x in tv])
        tva = tva.reshape((28,28,1))
    else :
        import skimage.data,skimage.transform
        tv = skimage.data.imread(filename)
        #print tv
        tva = [skimage.transform.resize(tv,(32,32))]
        tva = np.asarray(tva)
    return tva
def craftadv(x,y,sess,dataset='mnist'):
    assert dataset in ['mnist','roadsign']
    K.set_session(sess)
    if dataset == 'mnist':
        x = x.reshape((1,28,28,1))
        y = np.eye(10)[y]
        y = y.reshape(-1,10)
        model = load_model("model/model_mnist.h5")
    else :
        x = x.reshape((1,32,32,3))
        y = np.eye(62)[y]
        y = y.reshape(-1,62)
        model = load_model("model/model_traffic.h5")
    its, results = basic_iterative_method(
            sess, model, x, y, eps=0.4,
            eps_iter=0.010, clip_min=0.0,
            clip_max=1.0
    )
    x_adv = np.asarray([results[its[0], 0]])
    return x_adv
def cvs2png(x,dataset='mnist'):
    assert dataset in ['mnist','roadsign']
    fname = 'data/adv.png'
    x = np.rint(x*255.0)
    x = (x.astype(int)).clip(0,255)
    if dataset == 'mnist':
        im = Image.new('L',(28,28))
        im.putdata(x.flatten().tolist())
    else :
        im = Image.new('RGB',(32,32))
        x = list(tuple(i) for i in x.reshape(-1,3))
        im.putdata(x)
        #x = x.reshape(32,32,3)
        #im = Image.fromarray(x,'RGB')
        #imsave(fname,x.flatten())
    im.save(fname)
    return fname
def extract_lid(x,model,X):
    '''
    param: x ---- the x to estimated
    param: model ---- the model 
    param: X ---- the dataset (usually X_test)
    '''
    from learn import get_layer_wise_activations,mle_single
    x = x.reshape(1,28,28,1)
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
                 for out in get_layer_wise_activations(model, 'mnist')]
    lid = np.zeros(shape=(len(funcs))) 
    for i,func in enumerate(funcs):
        X_act = func([X,0])[0]
        x_act = func([x,0])[0]
        X_act = X_act.reshape(len(X_act),-1)
        x_act = x_act.reshape(1,-1)
        lid[i]=mle_single(X_act,x_act,k=10)
    lid = np.asarray(lid)
    print lid.shape

def calc_single_bim(x,sess):
    from scipy.spatial.distance import euclidean
    K.set_session(sess)
    model = load_model("model/model_mnist.h5")
    x = x.reshape(1,28,28,1)
    y = model.predict_classes(x)
    y = np.eye(10)[y]
    print y.shape
    its, results = basic_iterative_method(
            sess, model, x, y, eps=0.40,
            eps_iter=0.010, clip_min=-0.5,
            clip_max=0.5)
    x_adv = np.asarray([results[its[i], i] for i in range(len(y))])
    dist = np.zeros(shape=(1,1))
    dist[0][0] = euclidean(x.reshape(-1),x_adv.reshape(-1))
    return dist

if __name__=="__main__":
    import tensorflow as tf
    sess = tf.Session()
    K.set_session(sess)
    path = '/home/wlyu/cscn/presentation/data/mnist_png/mnist_png/testing/9/5233.png'
    x = png2cvs(path) 
    model = load_model("model/model_mnist.h5")
    dist =  calc_single_bim(x,model,sess)
    print dist
    print type(dist),dist.shape
