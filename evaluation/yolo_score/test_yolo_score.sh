cd "/home/wangph/layout2img"
export PYTHONPATH=$PWD

cd evaluation/yolo_score
echo $PWD


python test_yolo_score.py \
    --imageid_path image_id.txt \
    --image_path ../../outputs/tem_6/coco128_repeat1_thres2.0