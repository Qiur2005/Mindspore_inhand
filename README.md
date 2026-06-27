# MindSpore inhand 掌中宝
## 软件描述
软件集成了一些用mindspore训练的模型，包括图像分类、目标检测、语义分割等，用于展示mindspore的功能和性能。
## 软件功能一览

<p align = "center"> 
<img src="image.png" width="50%" height="50%" >
</p>

## 软件问题汇总
1. 问题1：软件只能在arm架构的手机上加载网络模型，在x86虚拟安卓机上无法加载。
   ![alt text](image-1.png)
2. 舞蹈梦工程无法使用，页面显示不齐全。
   ![alt text](image-2.png)
   该页面被设置为了横屏，但是页面是按竖屏设计的所以显示不全，只有半屏。更改后正常显示。
3. Quick Start内容网页都已经失效，打开后为空页面。
   <p align = "center">
   <img src="image-3.png" width="50%" height="50%" >
   </p>

## 新增功能：情绪识别

### 功能简介

基于 FER-2013 数据集训练的面部情绪识别模型，支持 7 种情绪的实时识别，模型经过 INT8 量化压缩后部署在 Android 端侧运行。

### 支持识别的情绪类别

| 类别 | 说明 |
|------|------|
| 😠 愤怒 | Angry |
| 😒 厌恶 | Disgust |
| 😨 恐惧 | Fear |
| 😊 开心 | Happy |
| 😢 悲伤 | Sad |
| 😲 惊讶 | Surprise |
| 😐 平静 | Neutral |

### 技术方案

- **数据集**：FER-2013（35,887 张 48×48 灰度人脸图像）
- **训练平台**：华为云 ModelArts
- **模型框架**：MindSpore
- **端侧部署**：MindSpore Lite，INT8 量化压缩
- **输入规格**：48×48 灰度图，归一化至 [-1, 1]
- **推理设备**：Android ARM 端侧 CPU

### 新增文件说明

```
app/
├── src/main/
│   ├── assets/
│   │   └── emotion_model.ms          # 量化后的端侧推理模型
│   ├── java/com/mindspore/himindspore/
│   │   └── emotion/
│   │       └── EmotionActivity.java  # 情绪识别主界面
│   └── res/layout/
│       └── activity_emotion.xml      # 情绪识别布局文件
```

### 使用方式

1. 打开 APP 主界面，点击右下角 **😊 情绪识别** 按钮
2. 进入情绪识别界面后，将面部对准摄像头
3. 点击 **拍照识别** 按钮
4. 界面显示识别结果（情绪类别 + 置信度百分比）





