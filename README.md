# Open spot detection

Count open spots per block from videos


## Getting Started

### Dependencies

* Python + pip

### Installing

```
pip3 install -r requirements.txt 
```

### Executing program

Follow these instructions to count the number of open spots ( left block and right block) on an .mp4 video.
- Run the script track.py ( Inside the folder Yolov5_DeepSort_Pytorch)
```
python track.py —source yourvideopath —img 640 — yolo_model pathtotheyolov5weights —save-dir savedirectory —save-txt —conf-thres 0.42
```

