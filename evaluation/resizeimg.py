from skimage import io, img_as_ubyte
from skimage.transform import resize as imresize
import pathlib
import os
import tqdm

img_path = 'datasets/coco/val2017'
resolution = 128
save_path = 'outputs/coco128'


if not os.path.exists(save_path):
    os.makedirs(save_path)

path = pathlib.Path(img_path)
files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

for i, f in tqdm.tqdm(enumerate(files), total=len(files)):
    f = str(f)
    f_name = f.split('/')[-1].split('.')[0].split('_')[-1]
    file_path = os.path.join(save_path, 'img{:06d}.png'.format(int(f_name)))
    img = io.imread(f)
    resize_img = imresize(img, (resolution, resolution))
    io.imsave(file_path, img_as_ubyte(resize_img))

