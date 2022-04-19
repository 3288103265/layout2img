cd "/home/wangph/layout2img"
export PYTHONPATH=$PWD
echo $PWD

python evaluation/yolo_score/sample.py \
  --dataset coco \
  --model_path /home/wangph/LAMA/pretrained_models/coco_128.pth \
  --sample_path outputs/LAMA-pretrained/ \
  -r 1 \
  --image_id_savepath evaluation/yolo_score/image_id.txt



cd evaluation/yolo_score
echo $PWD

python test_yolo_score.py \
    --imageid_path image_id.txt \
    --image_path ../../outputs/LAMA-pretrained/coco128_repeat1_thres2.0