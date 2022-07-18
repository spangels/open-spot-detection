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
python3 track.py --source ../BLOCK_1.mp4 --img 640 --yolo_model ../best.pt --save-dir ../test --save-txt --conf-thres 0.42
```

- The argument save-dir is to tell where you want to store the predictions file. The predictions are stored in a txt format
- The argument conf-thres is to set the minimum confidence to accept a prediction. To maximize the Precision on the spot-2 count, I set this parameter to 0.42

Then to process the predictions:

- Run the script count_video.py ( Home directory)

```
python count_video.py —video-file yourvideopath —prediction-file pathtotxtpredicitionsfile —min-persistance int —conf-thres float —iou-threshold float —save-file pathtosavetheoutputfile
```

- The min-persistance argument is a post processing argument to clean the predictions. It sets the minimum persistance of an object in term of number of successive frames to consider it in the prediction. For instance, if a spot-1 is detected only in one frame of the video it is probably not reliable enough to keep it in the final count.
- Other cleaning functions are applied by the class CatchErrors in the count_video.py file. For instance, if an object change suddenly its class ( and therefore its ID) for one or two frames and then goes back to its initial class. We discard the intermediate class. Or if an object is initally detected as an open spot but then it turns out to be an invalid spot, we discard the open spot from the final count.
- -save-file Argument is the path where you want to store the csv file of the counts. Spots on the left and spots on the right are counted on two different columns
