export PYTHONPATH="/home/wangph/layout2img"

echo $PWD
echo $PYTHONPATH
python scripts/train_app.py \
  --out_path outputs/base2_debug\
  --batch_size 4 \
  --gpu_ids 2 \

  

