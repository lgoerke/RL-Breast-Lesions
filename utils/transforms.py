import torch
import random
import numpy as np
import utils.utils as u
import utils.image_manipulation as im
import copy
from utils.datasets import print_without_img


def check_bbox(r, c, rn, cn, height, width):
    if r < 0:
        r = 0
    if r >= height:
        r = height - 2

    if c < 0:
        c = 0
    if c >= width:
        c = width - 2

    if r + rn >= height:
        rn = cn = height - r - 1

    if c + cn >= width:
        rn = cn = width - c - 1
        
    if rn <= 0:
        rn = cn = 1
    return r, c, rn, cn


def recover_sample(sample, entry, new_data):
    sample[entry] = new_data
    return sample


def get_center_row(bb_old, bb_new):
    if bb_new[2] == 0:
        print('bb_old', bb_old)
        print('bb_new', bb_new)
        raise ValueError('{}, {}'.format(bb_old, bb_new))
    return int((((bb_old[1] - bb_new[1]) + (1 / 2) * bb_old[3]) / bb_new[3]) * 224)


def get_center_col(bb_old, bb_new):
    if bb_new[3] == 0:
        print('bb_old', bb_old)
        print('bb_new', bb_new)
        raise ValueError('{}, {}'.format(bb_old, bb_new))
    return int((((bb_old[0] - bb_new[0]) + (1 / 2) * bb_old[2]) / bb_new[2]) * 224)

def quintuple_sample(sample, bbox_very_very_small, bbox_very_small, bbox_small, bbox_big):
    orig_bb = copy.copy(sample['bbox'])

    new_sample_very_very_small = sample.copy()
    new_sample_very_small = sample.copy()
    new_sample_small = sample.copy()
    new_sample_big = sample.copy()

    new_sample_very_very_small['bbox'] = bbox_very_very_small
    new_sample_very_very_small['obbox'] = bbox_very_very_small
    if bbox_very_very_small is None:
        new_sample_very_very_small['label'] = 0
    else:
        new_sample_very_very_small['label'] = 1
        new_sample_very_small['center_row'] = get_center_row(orig_bb, bbox_very_very_small) / 244
        new_sample_very_small['center_col'] = get_center_col(orig_bb, bbox_very_very_small) / 244
    
    new_sample_very_small['bbox'] = bbox_very_small
    new_sample_very_small['obbox'] = bbox_very_small
    if bbox_very_small is None:
        new_sample_very_small['label'] = 0
    else:
        new_sample_very_small['label'] = 1
        new_sample_very_small['center_row'] = get_center_row(orig_bb, bbox_very_small) / 244
        new_sample_very_small['center_col'] = get_center_col(orig_bb, bbox_very_small) / 244

    new_sample_small['bbox'] = bbox_small
    new_sample_small['obbox'] = bbox_small
    new_sample_small['label'] = 0
    if not (bbox_small is None):
        new_sample_small['center_row'] = get_center_row(orig_bb, bbox_small) / 244
        new_sample_small['center_col'] = get_center_col(orig_bb, bbox_small) / 244

    new_sample_big['bbox'] = bbox_big
    new_sample_big['obbox'] = bbox_big
    new_sample_big['label'] = 0
    if not (bbox_big is None):
        new_sample_big['center_row'] = get_center_row(orig_bb, bbox_big) / 244
        new_sample_big['center_col'] = get_center_col(orig_bb, bbox_big) / 244

    return [sample, new_sample_very_very_small, new_sample_very_small, new_sample_small, new_sample_big]

def multiply_sample(sample, bbox_list, cat_list):
    sample_list = []
#     if len(bbox_list) == 6:
#         print(bbox_list)
    for idx,bbox in enumerate(bbox_list):
        new_sample = sample.copy()
        new_sample['bbox'] = bbox
        new_sample['obbox'] = bbox
        new_sample['label'] = cat_list[idx]
        sample_list.append(new_sample)
        del new_sample
#     if len(bbox_list) == 6:
#         for s in sample_list:
#             print(s['bbox'])
    return sample_list

####################################################################################
## IMAGE TRANSFORMS                                                               ##
####################################################################################

class BboxCrop(object):
    """Crops and resizes the given PIL Image within the bbox.
    Args:
        targetsize: targetsize of resize

    """

    def __init__(self, targetsize):
        self.targetsize = targetsize

    def get_name(self):
        return 'BboxCrop'
        
    def __call__(self, sample_list):
        """
        Args:
            sample: image, bbox, label
        Returns:
            PIL Image: Cropped image
        """
        out = []
        # print('Start cropping',flush=True)
        # print_without_img(sample_list)
        for sample in sample_list:
            if sample['has_lesion'] == 1:
                # If it is known where the lesion is, draw random padding percentage
                # and increase bbox respectively
                r, c, rn, cn = u.get_nums_from_bbox(sample['bbox'])
                # print('from bbox',r,c,rn,cn,flush=True)
                # Check if I have run out of image
                r, c, rn, cn = check_bbox(r, c, rn, cn, sample['image_height'], sample['image_width'])
            else:
                rn = cn = np.random.randint(3, sample['image_width'] - 3)
                r = np.random.randint(1, sample['image_height'] - rn)
                c = np.random.randint(1, sample['image_width'] - cn)

                # Check if I have run out of image
                if r + rn > sample['image_height']:
                    rn = cn = sample['image_height'] - r
                if c + cn > sample['image_width']:
                    rn = cn = sample['image_width'] - c

                r, c, rn, cn = check_bbox(r, c, rn, cn, sample['image_height'], sample['image_width'])
            # print('====================',flush=True)
            # print('before resize',r,c,rn,cn,flush=True)
            # print('img size',sample['image'].shape,flush=True)
            # print('====================',flush=True)
            im_copy = np.copy(sample['image'])
            resized = im.resize(sample['image'], r, c, rn, cn, self.targetsize, 0)
            recovered = recover_sample(sample, 'image', resized)
            out.append(recovered)
            if recovered['image'] is None:
                print('r,c,rn,cn, targetsize', r, c, rn, cn, self.targetsize)
                print('bbox', sample['bbox'])
                print('obbox', sample['obbox'])
                print('im_copy', im_copy.shape)
                if resized is None:
                    print('resized is None')
                else:
                    print('resized shape', resized.shape)
            # except AttributeError:
            #     ValueError('Crop old {} {} {} {}, bbox {}, obbox {}, img h {}, img w {}'.format(r,c,rn,cn,u.get_nums_from_bbox(out[-1]['bbox']),u.get_nums_from_bbox(out[-1]['obbox']),out[-1]['image_height'],out[-1]['image_width']))
        # print('Stop cropping',flush=True)
        # print_without_img(sample_list)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(targetsize={})'.format(self.targetsize)
    
class BboxCropMult(object):
    """Crops and resizes the given PIL Image within the bbox.
    Args:
        targetsize: targetsize of resize

    """

    def __init__(self, targetsize):
        self.targetsize = targetsize

    def get_name(self):
        return 'BboxCropMult'
        
    def __call__(self, sample_list):
        """
        Args:
            sample: image, bbox, label
        Returns:
            PIL Image: Cropped image
        """
        out = []
        # print('Start cropping',flush=True)
        # print_without_img(sample_list)
        for sample in sample_list:
            # If it is known where the lesion is, draw random padding percentage
            # and increase bbox respectively
            try:
                r, c, rn, cn = u.get_nums_from_bbox(sample['bbox'])
            except:
                print_without_img(sample)
                raise ValueError('Whatever')
            # print('from bbox',r,c,rn,cn,flush=True)
            # Check if I have run out of image
            r, c, rn, cn = check_bbox(r, c, rn, cn, sample['image_height'], sample['image_width'])
            
            im_copy = np.copy(sample['image'])
            resized = im.resize(sample['image'], r, c, rn, cn, self.targetsize, 0)
            recovered = recover_sample(sample, 'image', resized)
            out.append(recovered)
            if recovered['image'] is None:
                print('r,c,rn,cn, targetsize', r, c, rn, cn, self.targetsize)
                print('bbox', sample['bbox'])
                print('obbox', sample['obbox'])
                print('im_copy', im_copy.shape)
                if resized is None:
                    print('resized is None')
                else:
                    print('resized shape', resized.shape)
            # except AttributeError:
            #     ValueError('Crop old {} {} {} {}, bbox {}, obbox {}, img h {}, img w {}'.format(r,c,rn,cn,u.get_nums_from_bbox(out[-1]['bbox']),u.get_nums_from_bbox(out[-1]['obbox']),out[-1]['image_height'],out[-1]['image_width']))
        # print('Stop cropping',flush=True)
        # print_without_img(sample_list)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(targetsize={})'.format(self.targetsize)

class Normalize(object):
    """Normalized the given PIL Image between 0 and 1
    Args:

    """    
    def get_name(self):
        return 'Normalize'

    def __call__(self, sample_list):
        """
        Args:
            sample: image, bbox, label
        Returns:
            PIL Image: Flipped image
        """
        out = []
        for sample in sample_list:
            normalized_img = sample['image']
            normalized_img = (normalized_img - normalized_img.min())/(normalized_img.max()-normalized_img.min())
            out.append(recover_sample(sample, 'image', normalized_img))
        return out

    def __repr__(self):
        return self.__class__.__name__


    
class RandomFlipImg(object):
    """Flips the given PIL Image
    Args:
        targetsize: targetsize of resize

    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def get_name(self):
        return 'RandomFlip'
        
    def __call__(self, sample_list):
        """
        Args:
            sample: image, bbox, label
        Returns:
            PIL Image: Flipped image
        """
        out = []
        for sample in sample_list:
            try:
                if np.random.rand() < self.prob:
                    flipped_img = np.array([np.fliplr(channel) for channel in sample['image']])
                else:
                    flipped_img = sample['image']
            except TypeError as err:
                raise TypeError("Error for this sample {} from sample list {} was {}".format(sample,sample_list,err))
            out.append(recover_sample(sample, 'image', flipped_img))
        return out

    def __repr__(self):
        return self.__class__.__name__


class RandomGammaImg(object):
    """Apply gamma transformation 
    Args:
        targetsize: targetsize of resize

    """

    def __init__(self, gamma_values=[0.8, 1.2], use_normal_distribution=False, prob=0.25):
        self.prob = prob
        self.gamma_values = gamma_values
        self.use_normal_distribution = use_normal_distribution
        if self.use_normal_distribution:
            self.gamma_max = max(self.gamma_values)
            self.gamma_min = min(self.gamma_values)
            self.std = (self.gamma_max - self.gamma_min) / 6.0  # 99.7% will be between min and max

    def get_name(self):
        return 'RandomGamma'
            
    def __call__(self, sample_list):
        """
        Args:
            sample: image, bbox, label
        Returns:
            PIL Image: Cropped image
        """
        out = []
        for sample in sample_list:
            try:
                if np.random.rand() < self.prob:
                    if self.use_normal_distribution:
                        gamma = np.clip(
                            np.random.normal(1.0, self.std),
                            self.gamma_min,
                            self.gamma_max
                        )
                    else:
                        gamma = random.choice(self.gamma_values)
                    gamma_img = np.power(sample['image'], gamma)

                else:
                    gamma_img = sample['image']
                out.append(recover_sample(sample, 'image', gamma_img))
            except:
                raise TypeError("Error for this sample {} from sample list {} was {}".format(sample,sample_list,sys.exc_info()[0]))
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(gamma values={},use normal distr={})'.format(self.gamma_values,
                                                                                        self.use_normal_distribution)


####################################################################################
## BOUNDING BOX TRANSFORMS                                                        ##
####################################################################################

class GetBBsMult(object):
    ''' Return a completely overlapping (DSC ~ 1), tightly overlapping (DSC ~ 0.6), a bit overlapping (DSC ~ 0.4) and nearly not overlapping (DSC ~ 0.2) bbox
    '''
    
    def get_name(self):
        return 'GetBBsMult'

    def __call__(self, sample_list):
        # print('Start quadrupling',flush=True)
        # print_without_img(sample_list)
        for sample in sample_list:
            if sample['has_lesion'] == 1:
                # print('-----')
                # print('bbox',sample['obbox'])
                r, c, rn, cn = u.get_nums_from_bbox(sample['bbox'])
                bbox_list = []
                factor_list = [4.15,2.92,2.59,\
                               2.33,2.13,1.97,1.83,\
                               1.71,1.6,1.51,1.42,\
                               1.35,1.28,1.21,1.15,\
                               1.09,1.04]
                cat_list = [0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
                for factor in factor_list:
                    row_lim = int(r - (factor-1) * rn)
                    if row_lim < 1:
                        row_lim = 1
                        if r <= 1:
                            r = 2
                    col_lim = int(c - (factor-1) * cn)
                    if col_lim < 1:
                        col_lim = 1
                        if c <= 1:
                            c = 2
                    try:
                        new_r = np.random.randint(row_lim, r)
                        new_c = np.random.randint(col_lim, c)
                    except ValueError:
                        raise ValueError(
                            'r {}, c {}, rn {}, cn {}, row_lim {}, col_lim {}'.format(r, c, rn, cn, row_lim, col_lim))

                    new_rn = int((factor) * rn)
                    new_cn = int((factor) * cn)
                    new_r, new_c, new_rn, new_cn = check_bbox(new_r, new_c, new_rn, new_cn, sample['image_height'],
                                                              sample['image_width'])
                    bbox_list.append(u.create_bbox(new_r, new_c, new_rn, new_cn))   
                bbox_list.append(sample['bbox'])
            else:
                bbox_list = []
                factor_list = [0.8,0.7,0.6,0.5,0.2,0.1]
                cat_list = [0,0,0,0,0,0]
                for factor in factor_list:
                    smaller_side = min(sample['image_height'], sample['image_width'])
                    rn = cn = int(smaller_side * factor)
                    r = np.random.randint(1, sample['image_height'] - rn)
                    c = np.random.randint(1, sample['image_width'] - cn)

                    # Check if I have run out of image
                    if r + rn > sample['image_height']:
                        rn = cn = sample['image_height'] - r
                    if c + cn > sample['image_width']:
                        rn = cn = sample['image_width'] - c

                    r, c, rn, cn = check_bbox(r, c, rn, cn, sample['image_height'], sample['image_width'])
                    bbox_list.append(u.create_bbox(r, c, rn, cn))

        return multiply_sample(sample, bbox_list,cat_list)

    
    
class GetFiveBBs(object):
    ''' Return a completely overlapping (DSC ~ 1), tightly overlapping (DSC ~ 0.6), a bit overlapping (DSC ~ 0.4) and nearly not overlapping (DSC ~ 0.2) bbox
    '''
    
    def get_name(self):
        return 'GetFiveBBs'

    def __call__(self, sample_list):
        # print('Start quadrupling',flush=True)
        # print_without_img(sample_list)
        for sample in sample_list:
            if sample['has_lesion'] == 1:
                # print('-----')
                # print('bbox',sample['obbox'])
                r, c, rn, cn = u.get_nums_from_bbox(sample['bbox'])

                ## Create even smaller new bbox
                row_lim = int(r - (1 / 4) * rn)
                if row_lim < 1:
                    row_lim = 1
                    if r <= 1:
                        r = 2
                col_lim = int(c - (1 / 4) * cn)
                if col_lim < 1:
                    col_lim = 1
                    if c <= 1:
                        c = 2
                try:
                    new_r = np.random.randint(row_lim, r)
                    new_c = np.random.randint(col_lim, c)
                except ValueError:
                    raise ValueError(
                        'r {}, c {}, rn {}, cn {}, row_lim {}, col_lim {}'.format(r, c, rn, cn, row_lim, col_lim))

                new_rn = int((5 / 4) * rn)
                new_cn = int((5 / 4) * cn)
                # print('Very very small before check',new_r,new_c,new_rn,new_cn)
                # print('height',sample['image_height'],'width',sample['image_width'])
                new_r, new_c, new_rn, new_cn = check_bbox(new_r, new_c, new_rn, new_cn, sample['image_height'],
                                                          sample['image_width'])
                # print('Very very small after check',new_r,new_c,new_rn,new_cn)
                new_bbox_very_very_small = u.create_bbox(new_r, new_c, new_rn, new_cn)
                
                ## Create smallest new bbox
                row_lim = int(r - (1 / 2) * rn)
                if row_lim < 1:
                    row_lim = 1
                    if r <= 1:
                        r = 2
                col_lim = int(c - (1 / 2) * cn)
                if col_lim < 1:
                    col_lim = 1
                    if c <= 1:
                        c = 2
                try:
                    new_r = np.random.randint(row_lim, r)
                    new_c = np.random.randint(col_lim, c)
                except ValueError:
                    raise ValueError(
                        'r {}, c {}, rn {}, cn {}, row_lim {}, col_lim {}'.format(r, c, rn, cn, row_lim, col_lim))

                new_rn = int((3 / 2) * rn)
                new_cn = int((3 / 2) * cn)
                # print('Very small before check',new_r,new_c,new_rn,new_cn)
                # print('height',sample['image_height'],'width',sample['image_width'])
                new_r, new_c, new_rn, new_cn = check_bbox(new_r, new_c, new_rn, new_cn, sample['image_height'],
                                                          sample['image_width'])
                # print('Very small after check',new_r,new_c,new_rn,new_cn)
                new_bbox_very_small = u.create_bbox(new_r, new_c, new_rn, new_cn)

                ## Create smaller new bbox
                row_lim = r - rn
                if row_lim < 1:
                    row_lim = 1
                    if r <= 1:
                        r = 2
                col_lim = c - cn
                if col_lim < 1:
                    col_lim = 1
                    if c <= 1:
                        c = 2
                try:
                    new_r = np.random.randint(row_lim, r)
                    new_c = np.random.randint(col_lim, c)
                except ValueError:
                    raise ValueError(
                        'r {}, c {}, rn {}, cn {}, row_lim {}, col_lim {}'.format(r, c, rn, cn, row_lim, col_lim))

                new_rn = 2 * rn
                new_cn = 2 * cn
                # print('Small before check',new_r,new_c,new_rn,new_cn)
                # print('height',sample['image_height'],'width',sample['image_width'])
                new_r, new_c, new_rn, new_cn = check_bbox(new_r, new_c, new_rn, new_cn, sample['image_height'],
                                                          sample['image_width'])
                # print('Small after check',new_r,new_c,new_rn,new_cn)
                new_bbox_small = u.create_bbox(new_r, new_c, new_rn, new_cn)

                ## Create larger new bbox
                row_lim = r - 2 * rn
                if row_lim < 1:
                    row_lim = 1
                    if r <= 1:
                        r = 2
                col_lim = c - 2 * cn
                if col_lim < 1:
                    col_lim = 1
                    if c <= 1:
                        c = 2

                try:
                    new_r = np.random.randint(row_lim, r)
                    new_c = np.random.randint(col_lim, c)
                except ValueError:
                    raise ValueError(
                        'r {}, c {}, rn {}, cn {}, row_lim {}, col_lim {}'.format(r, c, rn, cn, row_lim, col_lim))
                new_rn = 3 * rn
                new_cn = 3 * cn
                # print('Big before check',new_r,new_c,new_rn,new_cn)
                # print('height',sample['image_height'],'width',sample['image_width'])
                new_r, new_c, new_rn, new_cn = check_bbox(new_r, new_c, new_rn, new_cn, sample['image_height'],
                                                          sample['image_width'])
                # print('Big after check',new_r,new_c,new_rn,new_cn)
                new_bbox_big = u.create_bbox(new_r, new_c, new_rn, new_cn)
            else:
                new_bbox_very_very_small = None
                new_bbox_very_small = None
                new_bbox_small = None
                new_bbox_big = None
        # print('orig',sample['obbox'])
        # print('very small',new_bbox_very_small)
        # print('small',new_bbox_small)
        # print('big',new_bbox_big)
        return quintuple_sample(sample, new_bbox_very_very_small, new_bbox_very_small, new_bbox_small, new_bbox_big)

class RandomTranslateBB(object):
    """Translates the bbox in row and col direction (randomly)
    Returns:
        new_bbox: the translated bbox

    """
    def __init__(self, pixel_range, prob=0.5,cat=False):
        self.pixel_range = pixel_range
        self.prob = prob
        self.cat=cat

    def get_name(self):
        return 'RandomTranslate'
        
    def __call__(self, sample_list):
        """
        Args:
            sample: image, bbox, label
        Returns:
            list: new bbox
        """
        out = []
        # print('Start Translation',flush=True)
        # print_without_img(sample_list)
        for sample in sample_list:
            if self.cat or sample['has_lesion'] == 1:
                r, c, rn, cn = u.get_nums_from_bbox(sample['bbox'])
                if np.random.rand() < self.prob:
                    shift_r = np.random.randint(-self.pixel_range, self.pixel_range + 1)
                else:
                    shift_r = 0
                if np.random.rand() < self.prob:
                    shift_c = np.random.randint(-self.pixel_range, self.pixel_range + 1)
                else:
                    shift_c = 0
                # if np.random.rand() < self.prob:
                #     shift_r = -10
                #     shift_c = 7
                # else:
                #     shift_r = 0
                #     shift_c = 0

                new_r = r + shift_r
                new_c = c + shift_c

                new_r, new_c, new_rn, new_cn = check_bbox(new_r, new_c, rn, cn, sample['image_height'],
                                                          sample['image_width'])
                new_bbox = u.create_bbox(new_r, new_c, new_rn, new_cn)
            else:
                new_bbox = None
            out.append(recover_sample(sample, 'bbox', new_bbox))
            try:
                s = out[-1]['image'].shape
            except AttributeError:
                raise ValueError(
                    'Transform old {} {} {} {} new {} {} {} {}, bbox {}, obbox {}'.format(r, c, rn, cn, new_r, new_c,
                                                                                          new_rn, new_cn,
                                                                                          u.get_nums_from_bbox(
                                                                                              out[-1]['bbox']),
                                                                                          u.get_nums_from_bbox(
                                                                                              sample['obbox'])))
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(pixel_range={})'.format(self.pixel_range)


class RandomScaleBB(object):
    """Scales the bbox (randomly)
    Args:
        max_percentage: how many percent smaller or bigger the bbox is going to be maximal

    """

    def __init__(self, max_percentage, prob=0.5,cat=False):
        self.max_percentage = max_percentage
        self.prob = prob
        self.cat = cat

    def get_name(self):
        return 'RandomScale'
        
    def __call__(self, sample_list):
        """
        Args:
            sample: image, bbox, label
         Returns:
            list: new bbox
        """
        out = []
        for sample in sample_list:
            if sample['has_lesion'] == 1 or self.cat:
                if np.random.rand() < self.prob:
                    r, c, rn, cn = u.get_nums_from_bbox(sample['bbox'])
                    factor = np.random.uniform(1 - self.max_percentage, 1 + self.max_percentage)

                    # if self.max_percentage == 0.2:
                    #     factor = 1.1
                    # else:
                    #     factor = 0.9

                    new_rn = int(rn * factor)
                    new_cn = int(cn * factor)
                    diff_r = new_rn - rn
                    diff_c = new_cn - cn
                    # Get direction of scaling
                    rnd_num = np.random.rand()
                    # left up
                    if rnd_num < 0.2:
                        new_r = r - diff_r
                        new_c = c - diff_c
                    # left
                    elif rnd_num < 0.4:
                        new_r = r
                        new_c = c - diff_c
                    # up
                    elif rnd_num < 0.6:
                        new_r = r - diff_r
                        new_c = c
                    # right down
                    elif rnd_num < 0.8:
                        new_r = r
                        new_c = c
                    # center
                    else:
                        new_r = int(r - diff_r // 2)
                        new_c = int(c - diff_c // 2)
                    new_r, new_c, new_rn, new_cn = check_bbox(new_r, new_c, new_rn, new_cn, sample['image_height'],
                                                              sample['image_width'])
                    new_bbox = u.create_bbox(new_r, new_c, new_rn, new_cn)
                else:
                    new_bbox = sample['bbox']
            else:
                new_bbox = None
            out.append(recover_sample(sample, 'bbox', new_bbox))
            try:
                s = out[-1]['image'].shape
            except AttributeError:
                raise ValueError(
                    'Scale old {} {} {} {} new {} {} {} {}, bbox {}, obbox {}, img h {}, img w {}'.format(r, c, rn, cn,
                                                                                                          new_r, new_c,
                                                                                                          new_rn,
                                                                                                          new_cn,
                                                                                                          u.get_nums_from_bbox(
                                                                                                              out[-1][
                                                                                                                  'bbox']),
                                                                                                          u.get_nums_from_bbox(
                                                                                                              out[-1][
                                                                                                                  'obbox']),
                                                                                                          out[-1][
                                                                                                              'image_height'],
                                                                                                          out[-1][
                                                                                                              'image_width']))
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(max_percentage={})'.format(self.max_percentage)
