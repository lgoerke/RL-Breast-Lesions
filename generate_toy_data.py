import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from utils.noise import get_perlin_noise_img
from skimage import draw


# 12000 Train (3600 malignant)
# 2000 Val (600 malignant)

# 30% malignant

def sample_image_height_width(downsampling_factor):
    img_width = int(np.around(np.random.normal(566 // downsampling_factor, (136 / 2) // downsampling_factor)))
    if img_width < 200 // downsampling_factor:
        img_width = 200 // downsampling_factor

    img_height = int(np.around(np.random.normal(1081 // downsampling_factor, (139 / 2) // downsampling_factor)))
    if img_height < 500 // downsampling_factor:
        img_height = 500 // downsampling_factor
    return img_height, img_width


def create_negative_sample(idx, data_folder, val=False, with_noise=False):
    if with_noise:
        if val:
            filename = data_folder + 'png/neg_val_{}.png'.format(idx)
        else:
            filename = data_folder + 'png/neg_train_{}.png'.format(idx)

        downsampling_factor = 4
        img_height, img_width = sample_image_height_width(downsampling_factor)
        neg_img = get_perlin_noise_img(img_height, img_width)
        im = Image.fromarray(np.uint8(neg_img))
        im.save(filename)

        img_dict = {}
        if val:
            img_dict["file_name"] = 'neg_val_{}.png'.format(idx)
        else:
            img_dict["file_name"] = 'neg_train_{}.png'.format(idx)
        img_dict["height"] = img_height
        img_dict["width"] = img_width
        img_dict['id'] = idx

    else:
        img_dict = {"file_name": ('black.png',), "height": 1014, "width": 543, 'id': idx}

    return img_dict


def create_positive_sample(idx, data_folder, val=False, with_noise=False):
    if val:
        filename = data_folder + 'png/pos_val_{}.png'.format(idx)
    else:
        filename = data_folder + 'png/pos_train_{}.png'.format(idx)

    downsampling_factor = 4
    img_height, img_width = sample_image_height_width(downsampling_factor)

    bb_width = int(np.around(np.random.normal(84 // downsampling_factor, (53 / 2) // downsampling_factor)))
    if bb_width < 10 // downsampling_factor:
        bb_width = 10 // downsampling_factor
    elif bb_width > img_width:
        bb_width = img_width - 10 // downsampling_factor

    bb_height = int(np.around(np.random.normal(85 // downsampling_factor, (57 / 2) // downsampling_factor)))
    if bb_height < 10 // downsampling_factor:
        bb_height = 10 // downsampling_factor
    elif bb_height > img_height:
        bb_height = img_height - 10 // downsampling_factor

    row_lim = img_height - bb_height
    if row_lim < 1:
        row_lim = 1

    col_lim = img_width - bb_width
    if col_lim < 1:
        col_lim = 1

    bb_pos_row = np.random.randint(1, row_lim)
    bb_pos_col = np.random.randint(1, col_lim)

    if bb_width == 0 or bb_height == 0:
        print(bb_pos_col, bb_pos_row, bb_height, bb_width)
    if with_noise:
        pos_img = get_perlin_noise_img(img_height, img_width)
        # pos_img = np.zeros((img_height, img_width, 3))

        rr, cc = draw.ellipse(r=bb_pos_row + (1 / 2) * bb_height, c=bb_pos_col + (1 / 2) * bb_width,
                              r_radius=bb_height / 2, c_radius=bb_width / 2, shape=pos_img.shape)
        pos_img[rr, cc, :] = 255
        # pos_img[bb_pos_row:bb_pos_row+bb_height,bb_pos_col:bb_pos_col+bb_width,1] = 255

    else:
        pos_img = np.zeros((img_height, img_width, 3))
        pos_img[bb_pos_row:bb_pos_row + bb_height, bb_pos_col:bb_pos_col + bb_width, :] = 255

    im = Image.fromarray(np.uint8(pos_img))
    im.save(filename)

    img_dict = {}
    if val:
        img_dict["file_name"] = 'pos_val_{}.png'.format(idx)
    else:
        img_dict["file_name"] = 'pos_train_{}.png'.format(idx)
    img_dict["height"] = img_height
    img_dict["width"] = img_width
    img_dict['id'] = idx

    annotation_dict = {'bbox': [bb_pos_col, bb_pos_row, bb_width, bb_height], 'category_id': 2, 'id': idx,
                       'image_id': idx}

    return img_dict, annotation_dict


def main():
    data_folder = '/mnt/synology/breast/projects/lisa/toy_data/'
    # data_folder = '/Users/lisa/Documents/Uni/ThesisDS/local_python/toy_data/'
    # data_folder = '/Volumes/breast/projects/lisa/toy_data/'

    with_noise = True

    # filename = data_folder + 'png/black.png'
    # black_img = np.zeros((1014, 543, 3))
    # im = Image.fromarray(np.uint8(black_img))
    # im.save(filename)
    #
    # filename = data_folder + 'png/black_noise.png'
    # black_img = np.zeros((1014, 543, 3))
    # noise = np.random.beta(1, 2, size=black_img.shape[:2]) * 128
    # black_img[:, :, 0] += noise
    # black_img[:, :, 1] += noise
    # black_img[:, :, 2] += noise
    # black_img = np.clip(black_img, 0, 255)
    # im = Image.fromarray(np.uint8(black_img))
    # im.save(filename)

    coco_dict = {}
    coco_dict['type'] = 'instances'
    coco_dict['categories'] = []
    coco_dict['images'] = []
    coco_dict['annotations'] = []

    np.random.seed(4211)

    # Train malignant
    for idx in tqdm(range(1, 601)):
        img_dict, annotation_dict = create_positive_sample(idx, data_folder, with_noise=with_noise)
        coco_dict['images'].append(img_dict)
        coco_dict['annotations'].append(annotation_dict)

    # Train benign
    for idx in tqdm(range(601, 721)):
        img_dict = create_negative_sample(idx, data_folder, with_noise=with_noise)
        coco_dict['images'].append(img_dict)

    with open(data_folder + 'annotations/toy_coco_train.json', 'w') as fp:
        json.dump(coco_dict, fp)

    coco_dict = {'type': 'instances', 'categories': [], 'images': [], 'annotations': []}

    # Val malignant
    for idx in tqdm(range(1, 601)):
        img_dict, annotation_dict = create_positive_sample(idx, data_folder, val=True, with_noise=with_noise)
        coco_dict['images'].append(img_dict)
        coco_dict['annotations'].append(annotation_dict)

    # Val benign
    for idx in tqdm(range(601, 721)):
        img_dict = create_negative_sample(idx, data_folder, with_noise=with_noise)
        coco_dict['images'].append(img_dict)

    with open(data_folder + 'annotations/toy_coco_val.json', 'w') as fp:
        json.dump(coco_dict, fp)


if __name__ == '__main__':
    main()
    # im = get_perlin_noise_img(200, 150)
    # # print(im)
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(im)
    # plt.show()
    # data_folder = '/Users/lisa/Documents/Uni/ThesisDS/local_python/toy_data/'
    # create_positive_sample(1, data_folder, with_noise=True)
    # create_positive_sample(2, data_folder, with_noise=True)
    # create_positive_sample(3, data_folder, with_noise=True)

    # create_positive_sample(4, data_folder, with_noise=True)
    # create_positive_sample(5, data_folder, with_noise=True)
    # create_positive_sample(6, data_folder, with_noise=True)

    # create_positive_sample(7, data_folder, with_noise=True)
    # create_positive_sample(8, data_folder, with_noise=True)
    # create_positive_sample(9, data_folder, with_noise=True)

    # create_positive_sample(10, data_folder, with_noise=True)
    # create_positive_sample(11, data_folder, with_noise=True)
    # create_positive_sample(12, data_folder, with_noise=True)
