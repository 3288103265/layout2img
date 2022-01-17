export PYTHONPATH="/home/wangph/layout2img"

echo $PWD
echo $PYTHONPATH
python scripts/train_app.py \
  --out_path outputs/base2\
  --batch_size 36 \
  --gpu_ids 1,2,3,4 \

  

