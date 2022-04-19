cd "/home/wangph/layout2img"
export PYTHONPATH=$PWD
echo $PWD

python evaluation/yolo_score/sample.py \
  --dataset coco \
  --model_path outputs/tem_6/model/G_50.pth \
  --sample_path outputs/tem_6/ \
  -r 1 \
  --image_id_savepath evaluation/yolo_score/image_id.txt