import numpy as np
import pandas as pd
import json
import os
import sys
import random
import PIL
import torch
import tqdm
import cv2
import re
import argparse


CLASSES = ['spot-1','spot-2','invalidSpot-fireHydrant','invalidspot-coloredCurb','invalidSpot-entrance','fireHydrant']

class CatchErrors : 
    def __init__(self, iou_threshold=0.2, min_persistance=5, conf_thres=None):
        self.iou_threshold = iou_threshold
        self.min_persistance = min_persistance
        self.conf_thres = conf_thres
    def bbox_iou(self,box1, box2,  eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    
        b1_x1, b1_x2 = box1['bbox_left'] , box1['bbox_left'] + box1['bbox_w'] 
        b1_y1, b1_y2 = box1['bbox_top']  , box1['bbox_h'] + box1['bbox_top']
    
        b2_x1, b2_x2 = box2['bbox_left'] , box2['bbox_left'] + box2['bbox_w'] 
        b2_y1, b2_y2 = box2['bbox_top']  , box2['bbox_h'] + box2['bbox_top']

        # Intersection area
        inter = max(0, (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) ) * \
            max (0, (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)) )

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union

        return iou  # IoU
    
    def class_switch_anomalie(self,frames):
        anomalies = {}
        for n, frame in enumerate(frames[1:]):
            precedent_frame = frames[n]
            if frame['object_id'] != precedent_frame['object_id'] and 'invalidspot' in frame['label'].lower() :
                iou = self.bbox_iou(frame,precedent_frame)
                if iou>0.20:
                    anomalies[str(precedent_frame['object_id'])] = iou
        return anomalies
    def confidence_anomalie(self,frames):
        if not self.conf_thres:
            return {}
        else:
            anomalies = {}
            for n, frame in enumerate(frames):
                if frame['confidence'] < self.conf_thres:
                    anomalies[str(frame['object_id'])] = frame['confidence']
            return anomalies
    def persistance_anomalie(self,frames):
        anomalies = {str(frame['object_id']) : 0 for frame in frames}
        for frame in frames:
            anomalies[str(frame['object_id'])] += 1
        return {key : value  for key, value in anomalies.items() if value < self.min_persistance}
    
class CountSpots : 
    
    def __init__(self, video_path, txt_path, classes,min_persistance, iou_threshold,conf_thres,**kwargs):
        self.classes = classes
        self.video_path = video_path
        self.txt_path = txt_path
        self.preds = self.process_predictions()
        self.anomalie_detector = CatchErrors(iou_threshold=iou_threshold,min_persistance=min_persistance, conf_thres=conf_thres)
        
    
    def process_predictions(self):
        result = []
        with open(self.txt_path,'r') as f:
            for line in f.readlines():
                result.append( self.process_line( line ) )
        return result
    
    def get_dims_videos(self):
        vcap = cv2.VideoCapture(self.video_path) # 0=camera
        if vcap.isOpened(): 
            width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH )
            height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT )
        return width, height
    
    def bbox_iou(self, box1, box2,  eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    
        b1_x1, b1_x2 = box1['bbox_left'] , box1['bbox_left'] + box1['bbox_w'] 
        b1_y1, b1_y2 = box1['bbox_top']  , box1['bbox_h'] + box1['bbox_top']
    
        b2_x1, b2_x2 = box2['bbox_left'] , box2['bbox_left'] + box2['bbox_w'] 
        b2_y1, b2_y2 = box2['bbox_top']  , box2['bbox_h'] + box2['bbox_top']

        # Intersection area
        inter = max(0, (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) ) * \
            max (0, (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)) )

    # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union

        return iou  # IoU
    
    def process_line(self, prediction):
    
        digit_regex = re.compile('\d+')
        numbers = digit_regex.findall(prediction)
        label = ""
        for elm in self.classes:
            if elm in prediction:
                label = elm
                break
        if label == "":
            if 'spot' in prediction and not 'invalid' in prediction:
                label="spot-1"
            
        
        frame_id, object_id = numbers[0], numbers[1]
        bbox_left, bbox_top = numbers[2], numbers[3]
        width, height = numbers[4], numbers[5]
        confidence = float(f"0.{numbers[-1]}")
        dictionary =  {'frame_id':int(frame_id),'object_id':int(object_id), 'label' : label,
           'bbox_left': int(bbox_left), 'bbox_top':int(bbox_top),
           'bbox_w':int(width), 'bbox_h':int(height),'confidence':confidence}
        return dictionary
    def process_blocks(self,series):
        center_w, center_h = self.get_dims_videos()
        center_w, center_h = center_w/2, center_h/2
        preds_left, preds_right = [], []
        for pred in series:
            if (pred['bbox_left']+pred['bbox_w']/2) < center_w:
                preds_left.append(pred)
            else:
                preds_right.append(pred)
        return preds_left, preds_right
    def split_by_series_of_frames(self):
        res = []
        serie = []
        for pred in self.preds:
            if len(serie) == 0:
                serie.append(pred)
            else:
                if pred['frame_id']-serie[-1]['frame_id'] < 5:
                    serie.append(pred)
                else:
                    res.append(serie)
                    serie = []
        return res
    def __count__(self,serie):
        switch_anomalies, persistance_anomalies = self.anomalie_detector.class_switch_anomalie(serie), self.anomalie_detector.persistance_anomalie(serie)
        conf_anomalies = self.anomalie_detector.confidence_anomalie(serie)
        res = set([])
        anomalies = dict(switch_anomalies,**persistance_anomalies)
        anomalies = dict(anomalies,**conf_anomalies).keys()
        for frame in serie:
            if frame['label'] in ['spot-1','spot-2'] and str(frame['object_id']) not in anomalies:
                res.add(frame['object_id'])
        return len(res)
    def count_preds(self):
        series = self.split_by_series_of_frames()
        left_count, right_count = 0, 0
        for serie in series:
            left_block, right_block = self.process_blocks(serie)
            left_count += self.__count__(left_block)
            right_count += self.__count__(right_block)
        return left_count, right_count
            
def build_predictions_dataframe(prediction_folder, video_folder,classes=CLASSES,
                                iou_threshold=0.2,min_persistance=4,conf_thres=0.4):
    df = {'block_id':[],'left':[],'right':[]}
    files = [f for f in os.listdir(prediction_folder) if 'checkpoint' not in f]
    for file_name in files:
        blocks_path = prediction_folder + file_name
        blocks = os.listdir(blocks_path)
        for block in blocks:
            if os.path.isdir(block):
                continue
            video_path = video_folder + file_name + f"/{block[:-3]}mp4"
            txt_path = prediction_folder+ f"{file_name}/{block}"
            counter =  CountSpots(video_path=video_path, txt_path=txt_path,
                    classes= CLASSES, iou_threshold=0.2, min_persistance=4,conf_thres=conf_thres)
            left, right = counter.count_preds()
            df['block_id'].append(f"{file_name}-{block[:-4]}")
            df['left'].append(left)
            df['right'].append(right)
    return pd.DataFrame(df)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-folder', type=str, default='', help='path of the folder cont')
    parser.add_argument('--prediction-folder', type=str, default='', help='path of the folder cont')
    parser.add_argument('--min-persistance', type=int, default=4, help='Set the minimum persistance of an object in term of number of frames to consider it in the prediction')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.3, help='iou threshold')
    parser.add_argument('--save-file',type=str,default='',help='where to save the prediction')
    opt = parser.parse_args()
    return opt

def main(opt):
    video_folder, prediction_folder = opt.video_folder, opt.prediction_folder
    min_persistance, conf_thres = opt.min_persistance, opt.conf_thres
    iou_threshod = opt.iou_threshold
    save_file = opt.save_file
    df = build_predictions_dataframe(prediction_folder, video_folder,classes=CLASSES,
                                iou_threshold=0.2,min_persistance=4,conf_thres=conf_thres)
    df.to_csv(save_file,index=False)
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    
    
    
        
            
        