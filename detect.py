import pickle
def detect(A):
    detector = pickle.load(open("model/detector.sav",'rb'))
    A_label = detector.predict(A)
    return A_label
if __name__=="__main__":
    from util import png2cvs,calc_single_bim,craftadv
    from classify import classify
    import tensorflow as tf
    sess = tf.Session()
    path = '/home/wlyu/cscn/presentation/data/mnist_png/mnist_png/testing/9/5233.png'
    x = png2cvs(path) 
    y = classify(x)
    advx=craftadv(x,y,sess)
    dist =  calc_single_bim(x,sess)
    print detect(dist)
    dist =  calc_single_bim(advx,sess)
    print detect(dist)
