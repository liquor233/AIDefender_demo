#!/usr/bin/python
from __future__ import division
#from __future__ import print_function
from __future__ import unicode_literals
from future_builtins import *

import tensorflow as tf
import keras.backend as K

from PyQt4.QtGui import *
from PyQt4.QtCore import pyqtSlot as Slot
from PyQt4.QtCore import Qt,QString

from ui_mainwindow import Ui_MainWindow
from util import png2cvs,cvs2png,craftadv,calc_single_bim
from classify import classify
from detect import detect

class InitialWindow(QMainWindow,Ui_MainWindow):
	def __init__(self,parent=None):
       		super(InitialWindow, self).__init__(parent)
      	 	self.setupUi(self)
        	self.sess = tf.Session()
                # alignment 
                self.NormalPic.setAlignment(Qt.AlignCenter)
                self.AdvPic.setAlignment(Qt.AlignCenter)
                #signal-slot
                self.uploadButton.clicked.connect(self.get_image)
                self.craftButton.clicked.connect(self.craft_image)
		self.NormalDetect.clicked.connect(self.normal_detect)
		self.RightDetect.clicked.connect(self.right_detect)


        @Slot()
        def get_image(self):
            self.normfname = QFileDialog.getOpenFileName(self, 'Open file',filter="Image files (*.jpg *.gif *.png *.jpeg *.ppm)") 
            normpic = QPixmap(self.normfname)
            normpic = normpic.scaled(300,300,Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.NormalPic.setPixmap(normpic)
        
        @Slot()
        def normal_detect(self):
            #print self.normfname, type(self.normfname)
            dataset = str(self.DataSetChooser.currentText())
            self.normx = png2cvs(str(self.normfname),dataset)
            self.normy = classify(self.normx,dataset)
            label = 1
            if dataset == 'roadsign':
                checkbox = {18:'can not go in',7:'walkman',22:'Can\' t go in',32:'limit 70 kph',25:'large car',31:'double car rouble',28:'stop'}
                try:
                    b = checkbox[int(self.normy)]
                    message = 'this is a picture showing '+b
                except KeyError:
                    message = 'this is a picture showing roadsign number'+str(self.normy)+'\n'
            #dist = calc_single_bim(self.normx,self.sess)  
            #label = detect(dist)
            else:
                message = "This picture is number"+str(self.normy)+"\n"
            if label==0:
                message = message+"this is a normal picture"
            else :
                pass
            QMessageBox.information(self,"Classify Results",QString(message))
        
        @Slot()
        def craft_image(self):
            dataset = str(self.DataSetChooser.currentText())
            self.advx=craftadv(self.normx,self.normy,self.sess,dataset)
            self.advy=classify(self.advx,dataset)
            self.advfname=cvs2png(self.advx,dataset)           
            print "finish"
            advpic = QPixmap(self.advfname)
            advpic = advpic.scaled(300,300,Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self.AdvPic.setPixmap(advpic)

        @Slot()
        def right_detect(self):
            #dist = calc_single_bim(self.advx,self.sess)  
            #label = detect(dist)
            dataset = str(self.DataSetChooser.currentText())
            label = 0
            if dataset == 'roadsign':
                checkbox = {18:'can not go in',7:'walkman',22:'Can\' t go in',32:'limit 70 kph',25:'large car',31:'double car rouble',28:'stop'}
                try:
                    b = checkbox[int(self.advy)]
                    message = 'this is a picture showing '+b
                except KeyError:
                    message = 'this is a picture showing roadsign number'+str(self.advy)+'\n'
            else : message = "This picture is number"+str(self.advy)
            if label==0:
                message = "ERROR!!!!this is an adversarial picture\n"+message
            else :
                message = message+"this is an adversarial picture"
            QMessageBox.information(self,"Classify Results",QString(message))
            

if __name__ == "__main__":
	import sys
	app = QApplication(sys.argv)
    	form = InitialWindow()
	form.show()
	app.exec_()
