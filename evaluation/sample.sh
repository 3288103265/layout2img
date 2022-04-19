export PYTHONPATH=$PWD
echo $PYTHONPATH

python evaluation/sample_image_from_dataset.py \
    --model_path outputs/base2_2/model/G_25.pth \
    --out_path outputs/base2_2/samples_G25_1 \
    --save_gt \
    --device "cuda:0"