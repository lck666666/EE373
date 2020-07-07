# EE373大作业
## A:移动互联通信训练:（微信小程序及前端）
前端是一个有3个Tab的微信小程序。

Tab1: 图片上传
1.首先点击“请指定图片名称”输入框，输入想要上传的图片的命名。
2.然后点击“上传单张图片”按钮，选择拍照或在相册中选取单张图片，然后点击“完成”。
3.上传成功后会提示：“上传成功”。

Tab2: 人脸比对
1.点击“上传人脸图片”，选取两张想要比对的人脸照片。
2.上传成功后，会显示“是同一个人！”或“不是同一个人！”

Tab3: 人脸匹配
1.点击“上传图片”，拍照或选取一张想要上传的图片。
2.系统会与已经从Tab1上传过的人脸图片进行比对，若寻找到匹配的人脸，则会显示该人脸图片，框选出该人脸并显示之前通过Tab1对该图片的命名。若未寻找到，则提示：“没有匹配的照片”。


## B:移动互联数据的存储及计算训练（后端配置）
返回的json由三部分组成，’code’+’data’+’message’。
Code代表了错误的形式。<br> 
100代表前后端交互正确，无错误。<br> 
101代表请求参数错误，可能少了个文件啥的。<br> 
102代表发送的文件格式不正确。<br> 
103 代表发送的文件中存在识别不出人脸的图像。<br> 
104 代表数据库中没有相似的人脸。<br> 


首先是store这个api，路由地址为https://jingzhix.xyz:5000/store
需要接收一张照片 file1，如果请求时不发送任何数据，则返回的json数据如下：
```
{
  "code": 101,
  "data": {},
  "message": "Request parameter error"
}
```
如果发送的是png,jpg,jpeg以外的文件格式，则json数据如下,data中包含了服务器端图片存储的地址:
```
{
  "code": 102,
  "data": {},
  "message": "There is a malformed file"
}
```
如果正确上传图片文件，则返回json数据如下：
```
{
  "code": 100,
  "data": {},
  "message": "/home/ubuntu/xxx.jpg"
}
```
再是compare这个api，路由地址为：https://jingzhix.xyz:5000/face/compare
需要接收两张照片，分别为file1和file2
如果发送的两个文件中有识别不出来的人脸，则json数据如下:
```
{
  "code": 103,
  "data": {},
  "message": "There are unrecognized face images"
}
```
如果发送两个一样的人的照片，则json数据如下,Data中包含了face_distances和is_same_face这两个数据，代表两张照片中人脸的距离和是否为同一张人脸:

```
{
  "code": 100,
  "data":{
     "face_distances":[
        0.32323
     ],
     "is_same_face":[
        true
     ]
  },
  "message": ""
}
```
最后是face/recognition这个api, 路由地址为https://jingzhix.xyz:5000/face/recognition、
需要接收一张照片file1, 如果发送的照片在后端数据库中对应的人脸则返回的json数据如下，data中为对应人脸的名字：
```
{
  "code": 100,
  "data": "doreen1",
  "message": ""
}
```
如果找不到对应的人脸，则报错104:
```
{
  "code": 104,
  "data": {},
  "message": "There are no similar faces in the database"
}
```

## C:移动互联数据的实时交互与控制
### 直播环境的搭建：
...
### 基于Yolo的目标识别与分类
在服务器上安装 Darknet:
```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
然后选择相应的模型，由于服务器的配置比较低（1个CPU,2G内存),难以运行yolo3这样比较大的模型，所以我们在 cfg/ 中选择了网络规模较小的yolo2。
下载已经训练好的模型参数。
```
wget https://pjreddie.com/media/files/yolo.weights
```

然后将yolo的分类封装成 python.py,用于直播端进行调用。
在python.py中设置好默认的模型与参数后可直接进行使用，最后返回的是输出的预测结果与边界框的坐标。
```
Loading weights from /home/ubuntu/EE373/darknet/yolo.weights...Done!
[(b'bicycle', 0.8530982136726379, (341.83966064453125, 285.84002685546875, 492.8941650390625, 323.5599060058594)), 

(b'dog', 0.8239848613739014, (226.710205078125, 376.56317138671875, 189.13192749023438, 289.1216125488281)), 

(b'truck', 0.6359090805053711, (574.128173828125, 126.13597869873047, 212.53976440429688, 83.70970153808594))]
```
