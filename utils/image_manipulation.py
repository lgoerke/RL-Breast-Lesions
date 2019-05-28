import torch
import numpy as np
from skimage.transform import resize as sk_resize

from utils.torch_utils import numpy_to_tensor


###########################################################
# Image manipulations                                     #
###########################################################

def resize(im, r, c, rn, cn, crop_size=224, divide255=False):
    to_resize = im[:, r:r + rn, c:c + cn]
    if len(np.unique(to_resize)) > 0:
        # print(to_resize.shape)
        # print(type(to_resize))
        # resized_img = torch.zeros(1, 3, crop_size, crop_size)
        to_resize = numpy_to_tensor(
            np.reshape(to_resize, (1, to_resize.shape[0], to_resize.shape[1], to_resize.shape[2]))).float()
        resized_img = torch.nn.functional.interpolate(
            to_resize, size=(crop_size,crop_size), mode='bilinear', align_corners=False
        )

        # resized_img[0, 0, :, :] = torch.nn.functional.interpolate(
        #     to_resize[0, 0, :, :].view(1, 1, to_resize.shape[2], to_resize.shape[3]),
        #     size=(crop_size, crop_size), scale_factor=None,
        #     mode='bilinear', align_corners=False)
        # resized_img[0, 1, :, :] = torch.nn.functional.interpolate(
        #     to_resize[0, 1, :, :].view(1, 1, to_resize.shape[2], to_resize.shape[3]),
        #     size=(crop_size, crop_size), scale_factor=None,
        #     mode='bilinear', align_corners=False)
        # resized_img[0, 2, :, :] = torch.nn.functional.interpolate(
        #     to_resize[0, 2, :, :].view(1, 1, to_resize.shape[2], to_resize.shape[3]),
        #     size=(crop_size, crop_size), scale_factor=None,
        #     mode='bilinear', align_corners=False)

        # print(resized_img.shape)

        # resized_img = torch.from_numpy(sk_resize(to_resize, (3, crop_size, crop_size), order=3, mode='reflect')).float()

    elif to_resize.shape[1] == 0 or to_resize.shape[2] == 0:
        return None
    else:
        value = to_resize[0, 0, 0]
        resized_img = torch.ones(3, crop_size, crop_size) * value

    if divide255:
        resized_img = resized_img / 255
    return resized_img


def resize_np(im, r, c, rn, cn, crop_size=224, divide255=False):
    # BBOX: [col,row,col+col_no,row+row_no]
    # print(im.shape,flush=True)
    to_resize = im[:, r:r + rn, c:c + cn]

    # print(to_resize.shape,flush=True)
    # print(to_resize,flush=True)
    # print(np.unique(to_resize),flush=True)
    # print(len(np.unique(to_resize)),flush=True)
    # print('++++++',flush=True)
    # print('++++++',flush=True)
    # print('++++++',flush=True)
    # print('++++++',flush=True)
    # print('++++++',flush=True)
    # to_resize = to_resize.permute(2,0,1)
    if len(np.unique(to_resize)) > 0:
        resized_img = torch.from_numpy(sk_resize(to_resize, (3, crop_size, crop_size), order=3, mode='reflect')).float()
    elif to_resize.shape[1] == 0 or to_resize.shape[2] == 0:
        return None
    else:
        value = to_resize[0, 0, 0]
        resized_img = torch.ones(3, crop_size, crop_size) * value

    if divide255:
        resized_img = resized_img / 255
    return resized_img.unsqueeze(0)
