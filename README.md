# AIDefender_demo  
this is a demo written by zhanghao and liquor233 used for ciscn's sencond presentation  
(on the day **5.18**)  
## function  
our simple demo is about adversarial sample in machine learning area   
## 使用方法  
运行该demo,点击upload,可以上传mnist或者roadsign数据集(使用下拉选框进行选择,数据放在data文件夹中)  
选择左边的detect,发现分类正确  
选择craft,等待一定时间,可以看到右边加载出来了恶意的图片  
之后选择右边的detect,可以发现分类错误,说明攻击成功  
## 代码结构
.  
├── attacks.py(用于实施恶意样本攻击)  
├── classify.py(用于执行分类)  
├── craft\_adv\_examples.py(用于构造恶意样本)  
├── data(用于存放数据)  
│   ├── adv.png(构造的恶意图片)  
│   ├── mnist\_png  
│   ├── mnist.py  
│   ├── mnist\_readme.txt  
│   ├── mnist\_test\_10.csv  
│   ├── mnist\_train\_100.csv  
│   ├── test\_1.npy  
│   ├── test\_2.npy  
│   ├── test\_mnist1.npy    
│   ├── test\_mnist2.npy   
│   ├── traffic    
│   └── try.npy     
├── detect.py(用于检测恶意样本)        
├── learn.py(部分引用的别人的库函数)  
├── mainwindow.py(主ui界面)  
├── mainwindow.ui(主ui设计文件)  
├── model(模型存储目录,模型在云端训练好)  
│   ├── detector.sav(detector)  
│   ├── model\_mnist.h5  
│   └── model\_traffic.h5  
├── README.md  
├── traffic\_sign2.py(roadsign数据集模型训练文件)  
├── ui\_mainwindow.py(ui设计文件)  
├── util.py(某些工具库函数)  
## 写在后面
2天内和队友@zhanghao一起极限写demo,虽然很粗糙很简陋,但也是蛮不错的体验lalalalalala
