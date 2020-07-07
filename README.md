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
我们组选择使用python+flask的框架进行后端配置，详情参见附件中的test.py

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

然后将yolo的分类封装成 darknet.py,用于直播端进行调用，在darknet.py中，最重要的函数为detect函数，如下所示。
```Python
def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
```
在darknet.py中设置好默认的模型与参数后可直接进行使用(由于服务器性能原因，在我们组的实验中均适用yolo2进行图像处理），最后返回的是输出的预测结果与边界框的坐标。
```
Loading weights from /home/ubuntu/EE373/darknet/yolo.weights...Done!
[(b'bicycle', 0.8530982136726379, (341.83966064453125, 285.84002685546875, 492.8941650390625, 323.5599060058594)), 

(b'dog', 0.8239848613739014, (226.710205078125, 376.56317138671875, 189.13192749023438, 289.1216125488281)), 

(b'truck', 0.6359090805053711, (574.128173828125, 126.13597869873047, 212.53976440429688, 83.70970153808594))]
```
### 直播环境的搭建
搭建好直播环境后（组员利用了在别的课配置好的阿里云服务器），利用opencv内置的ffmpeg的包可以用来读取rtmp和rtsp流，在ipcamera.py中import我们yolo部分封装好的darknet.py，对视频流每10s进行一次抽帧后将图片交给darknet.py的detect函数，返回预测结果与框的位置，然后用ipcamera中定义的process_yolo函数将这些信息画在图片上, 关键部分的代码如下所示。
```Python
def process_yolo(yolo_out,img):
    l=len(yolo_out)
    for i in range(l):
        tmp=yolo_out[i];#相当于一个tuple like ('cat',0.84,(1,2,3,4))第一个为class第二个为概率第三个里面是bbox
        classes=str(tmp[0], encoding = "utf-8")  
        prob=tmp[1]
        bbox=tmp[2]
        img=addlayer(img,bbox,classes,prob)#将这些信息画在原始图片上
    return(img)
```
```Python
while cap.isOpened():
    success,frame = cap.read()
    if success:
        '''
		对frame进行识别处理
		'''
        frame_out=frame
        end_time=time.time()
        if end_time-start_time>2:#每过十秒读取一次
            
            n=n+1
            saving_path='/home/ubuntu/EE373/zhiboimg/'+str(n)+'.jpg'
            cv2.imwrite(saving_path,frame)#没高兴做直接的流处理而是每十秒保存一张图片
            # frame_out=frame
            r = darknet.detect(net, meta, saving_path.encode('utf-8'))
            # net=load_net("cfg/tiny-yolo.cfg","tiny-yolo.weights",0)
            # meta=load_meta("cfg/coco.data")
            # r=detect(net,meta,"zhiboimg/"+str(n)+'.jpg')
            # 运行yolo
            frame_out=process_yolo(r,frame)
            print(r)
            cv2.imwrite('/home/ubuntu/EE373/zhiboout/'+str(n)+'.jpg',frame_out)
            start_time=time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        
        pipe.stdin.write(frame_out.tostring())
```
### 小车的控制和避障
无人驾驶决策控制系统的任务就是根据给定的路网文件、获取的交通环境信息和自身行驶状态，将行为预测、路径规划以及避障机制三者结合起来，自主产生合理驾驶决策，实时完成无人驾驶动作规划。狭义上来讲，包含了无人驾驶车的行为决策、动作规划以及反馈控制模块；广义上来讲，还紧密依赖上游的路由寻径、交通预测模块的计算结果。
对于自动小车驾驶，我们仅从狭义考虑，可以大致分为三个阶段：
行为决策，任务是汇聚分析各种信息，做出行驶的决策，确定无人驾驶汽车应该进入什么行驶模式，比如路口左转模式、超车模式等，与上游模块信息相连。
动作规划，任务是将行为决策的宏观指令解释成一条带有时间信息的轨迹曲线，来给最底层的反馈控制来进行实际对车的操作；
反馈控制，任务是控制车辆尽可能遵循上游动作规划所输出的轨迹，通过控制方向盘转角以及前进速度实现。
行为决策
大概分成四个步骤：
1.	结合我车状态，地图数据，感知结果构建不同层次的场景。
2.	每个场景根据自身规则（交通法规，安全避让），计算出每个场景的个体决策。
3.	检查各个场景有无冲突，并解决（安全验证）。
4.	在统一的时空里，推演所有决策能否汇总成安全无碰的综合决策，最后发送给动作规划模块。
反馈控制（自动控制模块）
输入：为局部路径、车辆状态、车辆位置和终端命令
输出：为油门、刹车和转向等操作（一般来说，路径、时间、地点都是决策和规划层设计好的，控制层只要完成这些目标就可以）
基本控制内容：横向控制（MPC算法）和纵向控制（PID算法）
主要的控制命令输出：
```
// next id : 27

message ControlCommand {
  optional apollo.common.Header header = 1;
  // target throttle in percentage [0, 100]
  optional double throttle = 3;
  // target brake in percentage [0, 100]
  optional double brake = 4;
  // target non-directional steering rate, in percentage of full scale per
  // second [0, 100]
  optional double steering_rate = 6;
  // target steering angle, in percentage of full scale [-100, 100]
  optional double steering_target = 7;
  // parking brake engage. true: engaged
  optional bool parking_brake = 8;
  // target speed, in m/s
  optional double speed = 9;
  // target acceleration in m`s^-2
  optional double acceleration = 10;
  // model reset
  optional bool reset_model = 16 [deprecated = true];
  // engine on/off, true: engine on
  optional bool engine_on_off = 17;
  // completion percentage of trajectory planned in last cycle
  optional double trajectory_fraction = 18;
  optional apollo.canbus.Chassis.DrivingMode driving_mode = 19
      [deprecated = true];
  optional apollo.canbus.Chassis.GearPosition gear_location = 20;
  optional Debug debug = 22;
  optional apollo.common.VehicleSignal signal = 23;
  optional LatencyStats latency_stats = 24;
  optional PadMessage pad_msg = 25;
  optional apollo.common.EngageAdvice engage_advice = 26;
  optional bool is_in_safe_mode = 27 [default = false];
  // deprecated fields
  optional bool left_turn = 13 [deprecated = true];
  optional bool right_turn = 14 [deprecated = true];
  optional bool high_beam = 11 [deprecated = true];
  optional bool low_beam = 12 [deprecated = true];
  optional bool horn = 15 [deprecated = true];
  optional TurnSignal turnsignal = 21 [deprecated = true];
}
```
避障机制
1.	运动障碍物检测：对运动过程中环境中的运动障碍物进行检测,主要由车载环境感知系统完成。（很明显，从常识角度看，避开障碍物的第一步就是检测障碍物。）
2.	运动障碍物碰撞轨迹预测：对运动过程中可能遇到的障碍物进行可能性评级与预测,判断与无人驾驶车辆的碰撞关系。（当你检测到障碍物后，你就得让机器判断是否会与汽车相撞）
3.	运动障碍物避障：通过智能决策和路径规划,使无人驾驶车辆安全避障,由车辆路径决策系统执行。（判断了可能会与汽车发生碰撞的障碍物后，你就得去让机器做出决策来避障了）
前两项本质上是多目标识别和追踪，运动障碍物的避障本质上是一个路径规划的过程，在路段上有未知障碍物的情况下,按照一定的评价标准,寻找一条从起始状态到目标状态的无碰撞路径。

