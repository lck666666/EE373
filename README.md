# EE373大作业

## A:移动互联通信训练:（微信小程序及前端）
...

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
