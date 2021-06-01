# copied from: https://pytorch.org/vision/0.9/_modules/torchvision/utils.html#draw_bounding_boxes
# release since torchvision 0.9.0
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont, ImageColor 


@torch.no_grad()
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: int = 10,
    label_redlist: Optional[List[str]] = None
) -> torch.Tensor:

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If filled, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C x H x W)
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
            be represented as `str` or `Tuple[int, int, int]`.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
        label_redlist (List[str]): The label should not be drawn.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")

    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)

    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")

    else:
        draw = ImageDraw.Draw(img_to_draw)

    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

    for i, bbox in enumerate(img_boxes):
        
        if label_redlist:
            if labels[i] in label_redlist:
                continue
        
        if colors is None:
            color = None
        else:
            color = colors[i]

        if fill:
            if color is None:
                fill_color = (255, 255, 255, 100)
            elif isinstance(color, str):
                # This will automatically raise Error if rgb cannot be parsed.
                fill_color = ImageColor.getrgb(color) + (100,)
            elif isinstance(color, tuple):
                fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if labels is not None:
            draw.text((bbox[0], bbox[1]), labels[i], fill=color, font=txt_font)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)