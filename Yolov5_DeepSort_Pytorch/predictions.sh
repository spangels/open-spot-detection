for FILE in ../data/videos/*/* ; do videoPath="${FILE%/*}"; 
pattern="../data/videos/"; replace="../data/predictions_conf_0.42/";
save_dir="${videoPath/$pattern/$replace}";
args=(
    --source="$FILE"
    --img=640
    --yolo_model="../yolov5/runs/train/yolov5m-10pct-15k/weights/best.pt"
    --save-dir="$save_dir"
    --save-txt
    --conf-thres=0.42
            );
python track.py "${args[@]}" ; done ;