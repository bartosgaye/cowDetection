# Training YOLOv3 on a Custom Dataset

This guide provides a step-by-step workflow for training YOLOv3 to detect a custom object, in this case, a specific cow. The workflow combines both local and cloud-based computational resources for optimal performance.

---

## Table of Contents

* [Introduction](#introduction)  
* [Hardware and Software Environment](#hardware-and-software-environment)  
  * [Workflow Overview](#workflow-overview)  
* [Preprocessing Input Data](#preprocessing-input-data)  
* [Labeling Input Images](#labeling-input-images)  
  * [Manual Labeling with LabelImg](#manual-labeling-with-labelimg)  
  * [Semi-Automated Labeling using Object Tracking](#semi-automated-labeling-using-object-tracking)  
* [Configuring YOLOv3](#configuring-yolov3)  
  * [Download YOLOv3](#download-yolov3)  
  * [Modify YOLO Configuration](#modify-yolo-configuration)  
* [Training YOLOv3 on Google Colab](#training-yolov3-on-google-colab)  
* [References](#references)  

---

## Introduction

YOLO (You Only Look Once) is a state-of-the-art object detection model capable of identifying and localizing objects in images or video frames. Pre-trained versions are available for common objects (COCO dataset, 80 classes). In this project, we train YOLOv3 to detect a specific cow, but the workflow can be adapted for any object.

We use:

* **Darknet**: Open-source neural network framework for YOLO.
* **Google Colab**: Linux-based GPU environment for training.
* **Local machine**: Windows-based environment for preprocessing, manual labeling, and tracking using OpenCV, LabelImg, and dlib.

---

## Hardware and Software Environment

### Workflow Overview

1. **Image and Video Preprocessing**

   * Done locally using OpenCV and FFmpeg.
   * Includes resizing, frame extraction, format conversion, etc.

2. **Manual or Semi-Automated Labeling**

   * Use LabelImg for manual labeling.
   * Semi-automated labeling: Track objects in video with dlib, save bounding boxes as YOLO labels.

3. **Training and Testing YOLOv3**

   * Conducted on Google Colab with GPU support.
   * Custom anchor sizes calculated and YOLOv3 configured for your dataset.

4. **Interoperability**

   * Data is exchanged between local machine and Google Drive to manage preprocessing, training, and post-processing.

---

## Preprocessing Input Data

YOLO requires both images and corresponding `.txt` label files. Each line in the label file contains:

```

<class_id> <center_x> <center_y> <width> <height>

````

* `<class_id>`: Object class (0 if single class)  
* `<center_x>` & `<center_y>`: Center of bounding box relative to image dimensions  
* `<width>` & `<height>`: Width and height relative to image dimensions  

---

## Labeling Input Images

### Manual Labeling with LabelImg

Steps:

1. Take snapshots of the target cow from video using FFmpeg:

```bash
ffmpeg -i input_video.mp4 -vf "fps=0.5,scale=1920:-1" -q:v 1 snapshot%d.png
````

2. Label snapshots in **LabelImg**:

```bash
# Navigate to LabelImg folder
cd C:\your_path
# Run LabelImg
python labelImg.py
```

* Draw bounding boxes and assign class.
* Save annotations (`.txt`) for YOLO.
* If LabelImg assigns a different class ID (e.g., 15), adjust it to 0 manually or via provided script.

### Semi-Automated Labeling using Object Tracking

1. Select a clear frame of the target object and manually annotate the bounding box.
2. Use **dlib correlation tracker** to track the object in video.
3. Save frames and bounding boxes at intervals, converting coordinates from dlib format to YOLO format.
4. Save snapshots and `.txt` labels with consistent filenames.

---

## Configuring YOLOv3

### Download YOLOv3

Download Darknet YOLOv3 from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

### Modify YOLO Configuration

* `.cfg` file:

  * `batch = 64`
  * `subdivisions = 64`
  * `max_batches = 2000` (for 1 class)
  * `classes = 1`
  * `filters = (classes + 5) * 3 = 18`
  * Update `[yolo]` layers with calculated anchors.

* Custom anchors (from k-means clustering):

```
20 27, 16 23, 16 31, 18 27, 15 21, 19 26, 15 27, 15 25, 13 23
```

* `obj.data`:

```text
classes = 1
train = /content/darknet/train.txt
valid = /content/darknet/valid.txt
names = /content/darknet/obj.names
backup = /content/darknet/backup/
```

* `obj.names`:

```text
cow13
```

* `train.txt` / `valid.txt`:

```
/content/darknet/data/train/image1.jpg
/content/darknet/data/train/image2.jpg
...
```

---

## Training YOLOv3 on Google Colab

Upload the following files to Google Drive:

* `backup` folder
* `obj.data`
* `obj.names`
* `train.txt`
* `valid.txt`

Run in Colab:

```python
# Add execute permission
!chmod +x /content/drive/MyDrive/darknet/darknet

# Change directory
%cd /content/drive/MyDrive/darknet

# Start training
!./darknet detector train /content/drive/MyDrive/darknet/data/obj.data /content/drive/MyDrive/darknet/cfg/yolov3.cfg -dont_show
```

* Save trained weights to `backup` folder.
* Test detection on a video:

```bash
!./darknet detector demo /content/drive/MyDrive/darknet/data/obj.data \
/content/drive/MyDrive/darknet/cfg/yolov3.cfg \
/content/drive/MyDrive/darknet/backup/yolov3_best.weights \
-dont_show -out /content/drive/MyDrive/outputVideo.mp4 \
/content/drive/MyDrive/cutVideo.mp4
```

* Detect on a single image:

```bash
!./darknet detector test /content/drive/MyDrive/darknet/data/obj.data \
/content/drive/MyDrive/darknet/cfg/yolov3.cfg \
/content/drive/MyDrive/darknet/backup/yolov3_best.weights \
/content/drive/MyDrive/cow3.png -dont_show -out /content/drive/MyDrive/outputImage.jpg
```

---

## References

1. Redmon, J., Divvala, S., Girshick, R., Farhadi, A., *You Only Look Once: Unified, Real-Time Object Detection*, 2016.
2. Lin, T.Y., et al., *Microsoft COCO: Common Objects in Context*, 2014.
3. [cowDetection GitHub Repository](https://github.com/bartosgaye/cowDetection), 2024.
4. [FFmpeg](https://www.ffmpeg.org/download.html)
5. [LabelImg](https://pypi.org/project/labelImg/1.4.0/)
6. [AlexeyAB Darknet](https://github.com/AlexeyAB/darknet)

