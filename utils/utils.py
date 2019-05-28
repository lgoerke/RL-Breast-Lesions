import numpy as np
import torch
import subprocess
import os
import gc
import datetime


###########################################################
# Other                                                   #
###########################################################

def calc_dice_np(im1, im2):
    # Adapted from https: // gist.github.com / brunodoamaral / e130b4e97aa4ebc468225b7ce39b3137
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 {} and im2 {} must have the same shape.".format(im1.shape, im2.shape))

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def calc_dice_np_specs(bbox1, bbox2):
    row1, col1, row_nums1, col_nums1 = get_nums_from_bbox(bbox1)
    row2, col2, row_nums2, col_nums2 = get_nums_from_bbox(bbox2)
    dr = min(row1 + row_nums1, row2 + row_nums2) - max(row1, row2)
    dc = min(col1 + col_nums1, col2 + col_nums2) - max(col1, col2)
    if (dr >= 0) and (dc >= 0):
        intersection = dr * dc
    else:
        return 0
    im_sum = row_nums1 * col_nums1 + row_nums2 * col_nums2
    if im_sum == 0:
        return 0
    return 2. * intersection / im_sum


def create_bbox(r, c, rn, cn):
    return [c, r, cn, rn]


def get_center_points(bbox):
    r, c, rn, cn = get_nums_from_bbox(bbox)
    point = np.zeros((1, 2))
    point[0, 0] = int(r + (1 / 2) * rn)
    point[0, 1] = int(c + (1 / 2) * cn)
    return point


def convert_bbs_to_points(bbox):
    if bbox is None:
        return None
    r, c, rn, cn = get_nums_from_bbox(bbox)
    points = np.zeros((4, 2))
    points[0, 0] = points[1, 0] = r
    points[0, 1] = points[2, 1] = c
    points[2, 0] = points[3, 0] = r + rn
    points[1, 1] = points[3, 1] = c + cn
    return points


def get_nums_from_bbox(bbox, correct_to_pos=True):
    '''
    Get individual numbers of bounding box
    Args:
        bbox: bounding box with [x (col), y (row), width (col_no), height (row_no)]

    Returns:
        row, col, row_nums, col_nums
        (y, x, height, width)
    '''
    r = int(bbox[1])
    c = int(bbox[0])
    rn = int(bbox[3])
    cn = int(bbox[2])
    if correct_to_pos:
        if r < 0:
            r = 0
        if c < 0:
            c = 0
    return r, c, rn, cn


def get_mask_from_bb(image, bbox):
    mask = torch.zeros(image.shape[1], image.shape[2])
    mask[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2])] = 1
    return mask


# start = datetime.datetime.now()
# >>> end = datetime.datetime.now()

# >>> end - start
# datetime.timedelta(0, 3, 519319)


def memory():
    return int(open('/proc/self/statm').read().split()[1])


def print_current_time_epoch(start_time, epoch):
    total_time = datetime.datetime.now() - start_time
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', flush=True)
    print('Epoch {}, {:.0f} minutes passed ({:.1f} h).'.format(epoch, total_time.total_seconds() / 60,
                                                               (total_time.total_seconds() / 60) / 60), flush=True)
    print('Using {} MB GPU.'.format(get_memory_usage()), flush=True)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', flush=True)


def get_memory_usage():
    if torch.cuda.is_available():
        try:
            return get_gpu_memory_map()[0]
        except TypeError:
            return memory()
    else:
        return 0


def memory_usage():
    if torch.cuda.is_available():
        try:
            print('Using {} MB GPU.'.format(get_gpu_memory_map()), flush=True)
        except TypeError:
            print('Using {} GPU.'.format(memory()), flush=True)


class TimeTracker(object):
    def __init__(self):
        self.current_mb = 0
        self.action_before = "-"
        self.compare_action = "One step"
        self.compare_mb = 0
        self.running_mb = 0

    def stop_measuring_time(self, start_time, action, print_gpu_usage=False):
        # TODO comment in
        #     total_time = datetime.datetime.now() - start_time
        #     millisecs = total_time.total_seconds() * 1000
        #     print('{} took {:.0f} milliseconds = {:.0f} seconds = {:.0f} minutes.'.format(action, millisecs,
        #                                                                               total_time.total_seconds(),
        #                                                                               total_time.total_seconds() / 60),
        #       flush=True)

        #         if torch.cuda.is_available():
        #             try:
        #                 mb = get_gpu_memory_map()[0]

        #                 if not mb == self.current_mb:
        #                     print('== Between {} and {} GPU {} MB (total {}).'.format(self.action_before,action, mb - self.current_mb, mb), flush=True)
        # #                     self.running_mb += (mb - self.current_mb)

        #                 elif action == self.compare_action:
        #                     print('== Between {} and {} GPU {} MB (total {}).'.format(self.action_before,action, mb - self.current_mb, mb), flush=True)
        # #                     self.running_mb += (mb - self.current_mb)
        # #                 if action == self.compare_action:
        # #                     print('Overall mb difference {}'.format(self.running_mb - self.compare_mb), flush=True)
        # #                     self.compare_mb = mb
        #                 self.current_mb = mb

        #             except TypeError:
        #                 mb = memory()
        #                 if not mb == self.current_mb:
        #                     print('== Between {} and {} size {} MB (total {}).'.format(self.action_before,action, mb - self.current_mb, mb), flush=True)
        #                 self.current_mb = mb
        # #                     self.running_mb += (mb - self.current_mb)

        # #                 if action == self.compare_action:
        # #                     print('Overall mb difference {}'.format(self.running_mb - self.compare_mb), flush=True)
        # #                     self.compare_mb = mb

        #         self.action_before = action
        pass

    def start_measuring_time(self):
        return datetime.datetime.now()
        # return time.time()


def get_gc_size():
    cnt_objs = 0
    cnt = 0
    for obj in gc.get_objects():
        cnt_objs += 1
        try:
            mul = 1
            for e in obj.size():
                mul *= e
            cnt += mul
        # if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        except:
            pass
    return cnt, cnt_objs


def get_gpu_memory_map():
    """Get the current gpu usage.
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_paths(rsyncing, toy=False, notebook=False):
    """

    :param rsyncing:
    :param toy:
    :param notebook:
    :return:
    """
    if rsyncing:
        root = "/input"
    else:
        if toy:
            root = '/mnt/synology/breast/projects/lisa/toy_data/'
            # TODO delete this line on cluster 
            # root = '/Users/lisa/Documents/Uni/ThesisDS/local_python/toy_data'
            # root = '/Volumes/breast/projects/lisa/toy_data/'
        elif notebook:
            root = '/Volumes/breast/archives/screenpoint3'
        else:
            root = '/mnt/synology/breast/archives/screenpoint3'
    if toy:
        png_path = os.path.join(root, 'png')
    elif rsyncing:
        png_path = os.path.join(root, 'png')
    else:
        png_path = os.path.join(root, 'processed_dataset/png')
        #/mnt/synology/breast/archives/screenpoint3/processed_dataset/png

    if toy:
        anno_path_train = os.path.join(root, 'annotations/mscoco_train_full.json')
        anno_path_val = os.path.join(root, 'annotations/mscoco_val_full.json')
    elif notebook:
        anno_path_train = os.path.join(root, 'mscoco_train_flat.json')
        anno_path_val = os.path.join(root, 'mscoco_val_flat.json')
    else:    
        anno_path_train = '/mnt/synology/breast/archives/screenpoint3/mscoco_train_flat.json'
        anno_path_val = '/mnt/synology/breast/archives/screenpoint3/mscoco_val_flat.json'

    # # TODO delete next 3 lines
    # anno_path_train = '/Volumes/breast/archives/screenpoint3/mscoco_annotations/mscoco_train.json'
    # anno_path_val = '/Volumes/breast/archives/screenpoint3/mscoco_annotations/mscoco_val.json'
    # png_path = '/Volumes/breast/archives/screenpoint3/processed_dataset/png'

    # anno_path_train = '/Users/lisa/Documents/Uni/ThesisDS/local_python/real_data/annotations/mscoco_train.json'
    # anno_path_val = '/Users/lisa/Documents/Uni/ThesisDS/local_python/real_data/annotations/mscoco_val.json'
    # png_path = '/Users/lisa/Documents/Uni/ThesisDS/local_python/real_data/png'

    print('Train path is {}'.format(anno_path_train), flush=True)
    print('Validation path is {}'.format(anno_path_val), flush=True)
    print('Png path is {}'.format(png_path), flush=True)
    return anno_path_train, anno_path_val, png_path
