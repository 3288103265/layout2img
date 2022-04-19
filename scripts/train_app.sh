# export PYTHONPATH="/home/wangph/layout2img"

# echo $PWD
# echo $PYTHONPATH
# python scripts/train_app.py \
#   --out_path outputs/base2\
#   --batch_size 36 \
#   --gpu_ids 1,2,3,4 \

  
export PYTHONPATH="/home/wangph/layout2img"

echo $PWD
echo $PYTHONPATH
python scripts/train_app.py \
  --out_path outputs/base2_2\
  --batch_size 36 \
  --ckpt_from outputs/base2 \
  --gpu_ids 4,5,6,7 \
