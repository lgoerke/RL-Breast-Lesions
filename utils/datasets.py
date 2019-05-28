import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
from PIL import Image
import os
import utils.utils as u
import json
import math

class StratifiedSampler(torch.utils.data.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def get_side(filename):
    """
    Get information of side of breast on image from filename
    Args:
        filename of respective mammography image
    Returns:
        -1 for left breast (with attachment to left side, symbolic |>|)
        1 for right breast (with attachment to right side, symbolic |<|)
    """
#     name, extension = filename.split('.')
#     img_name = name.split('_')
#     img_name = img_name[0]
    img_name = filename
    if img_name[-1] == 'l':
        return -1
    elif img_name[-1] == 'r':
        return 1
    else:
        return None


def mammo_collate(batch):
    """
    Adapted from https://github.com/DIAGNijmegen/breast-mammo-lesion-detection/blob/master/src/common/data/mammo.py
    Puts each data field into a object with outer dimension batch size
    In a dictionary adds `_batch` to the key.
    """
    # print('Batch',batch)
    # The keys which can be collated
    mammo_keys = ['image', 'bbox', 'label', 'image_height', 'image_width', 'image_id', 'original', 'obbox',
                  'has_lesion', 'center_row', 'center_col']

    out = {}

    image_collate = []
    bbox_collate = []
    label_collate = []
    height_collate = []
    width_collate = []
    id_collate = []
    original_collate = []
    obbox_collate = []
    lesion_collate = []
    center_row_collate = []
    center_col_collate = []
    side_collate = []
    for sample_list in batch:
        # print('Sample List',sample_list)
        if sample_list:
            for d in sample_list:
                # print('d',d)
                if d['image'] is not None:
                    if len(d['image'].shape) > 3:
                        image_collate.append(np.squeeze(d['image']))
                    else:
                        image_collate.append(d['image'])
                    bbox_collate.append(d['bbox'])
                    label_collate.append(d['label'])
                    height_collate.append(d['image_height'])
                    width_collate.append(d['image_width'])
                    id_collate.append(d['image_id'])
                    original_collate.append(d['original'])
                    obbox_collate.append(d['obbox'])
                    lesion_collate.append(d['has_lesion'])
                    center_row_collate.append(d['center_row'])
                    center_col_collate.append(d['center_col'])
                    side_collate.append(d['side'])

    # print('Final len',len(lesion_collate))
    # print('Has lesions?',lesion_collate)
    # print('Labels',label_collate)

    out['image_batch'] = image_collate
    out['bbox_batch'] = bbox_collate
    out['label_batch'] = label_collate
    out['image_height_batch'] = height_collate
    out['image_width_batch'] = width_collate
    out['image_id_batch'] = id_collate
    out['original_batch'] = original_collate
    out['obbox_batch'] = obbox_collate
    out['has_lesion_batch'] = lesion_collate
    out['center_row_batch'] = center_row_collate
    out['center_col_batch'] = center_col_collate
    out['side_batch'] = side_collate

    return out
    # else:
    #     return default_collate(batch)


def get_balanced_weights_fixed_size(dataloader, size=1500):
    sample_me = []
    cnt_mal = 0
    other_cnt = 0
    cnt_mal_total = 0
    cnt_other_total = 0
    for batch in dataloader:
        for lbl in batch['label_batch']:
            if lbl == 1:
                if cnt_mal <= size//2:
                    sample_me.append(1)
                    cnt_mal +=1 
                else:
                    # TODO
                    sample_me.append(0)
#                     sample_me.append(1)
                cnt_mal_total += 1
            elif lbl == 0:
                if other_cnt <= size//2:
                    #TODO 
#                     sample_me.append(0)
                    sample_me.append(1)
                    other_cnt += 1
                else:
                    sample_me.append(0)
                cnt_other_total += 1
    print('Label lesion: {}\nNo label lesion:  {}'.format(cnt_mal_total,cnt_other_total))
    print('Sample Label lesion: {}\nSample no label lesion:  {}'.format(cnt_mal,other_cnt))
    return torch.Tensor(np.array(sample_me))


def get_balanced_weights(dataloader):
    is_malignant = []
    for batch in dataloader:
        for lbl in batch['label_batch']:
            if lbl == 1:
                is_malignant.append(1)
            else:
                is_malignant.append(0)
    total = len(is_malignant)
    malignant = np.sum(is_malignant)
    other = total - malignant

    weights = torch.zeros(total)
    for i, w in enumerate(weights):
        if is_malignant[i] == 1:
            weights[i] = total / malignant
        else:
            weights[i] = total / other

    return weights


def predictions_to_weights(predictions, labels):
    weights = torch.zeros(predictions.size())
    for idx, pred in enumerate(predictions):
        if labels[idx] == 1:
            weights[idx] = 1
        else:
            weights[idx] = pred
    return weights


def print_without_img(sample_list):
    for sample in sample_list:
        print('--~~~~~~~~~~~~~~~--', flush=True)
        print('image_height', sample["image_height"], flush=True)
        print('image_width', sample["image_width"], flush=True)
        print('image_id', sample["image_id"], flush=True)
        if sample['obbox'] is None and sample['bbox'] is None:
            print('Obbox and Bbox is None',flush=True)
        elif sample['obbox'] is None:
            print('Obbox is None',flush=True)
            print('bbox', u.get_nums_from_bbox(sample["bbox"]), flush=True)
        elif sample['bbox'] is None:
            print('BBox is None',flush=True)
            print('obbox', u.get_nums_from_bbox(sample["obbox"]), flush=True)
        else:
            print('obbox', u.get_nums_from_bbox(sample["obbox"]), flush=True)
            print('bbox', u.get_nums_from_bbox(sample["bbox"]), flush=True)
        print('label', sample["label"], flush=True)
        print('has_lesion', sample["has_lesion"], flush=True)
        if sample['image'] is None:
            print('image is None', flush=True)
        else:
            print('image is NOT None', flush=True)
            print('image shape', sample['image'].shape, flush=True)
        if sample['original'] is None:
            print('original is None', flush=True)
        else:
            print('original is NOT None', flush=True)
            print('original shape', sample['original'].shape, flush=True)
        print('-------------------', flush=True)


def get_anno_ids(annFile):
    dataset = json.load(open(annFile, 'r'))
    anno_ids = []
    for _ in dataset['annotations']:
        anno_ids.append(_['image_id'])

    return anno_ids


class dataset_coco(torch.utils.data.Dataset):
    """`

    Adapted from Torch vision CocoDetection
    MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, bbox_transform=None, label_transform=None, for_feature = False, add_border=False,cat=False,f_one=False):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        # self.ids = list(self.coco.imgs.keys())
        if for_feature:
            annotated_imgs = get_anno_ids(annFile)
            all_imgs = list(self.coco.imgs.keys())
            unannotated_imgs = list(set(annotated_imgs).symmetric_difference(set(all_imgs)))
            if cat:
                divisor = 3
            else:
                divisor = 5
            
            
            self.ids = annotated_imgs + unannotated_imgs[:((len(annotated_imgs)//divisor))]
            
            print(len(annotated_imgs),'annotations found',flush=True)
            print(len(all_imgs),'images found',flush=True)
        else:
            if f_one:
                annotated_imgs = get_anno_ids(annFile)
                all_imgs = list(self.coco.imgs.keys())
                unannotated_imgs = list(set(annotated_imgs).symmetric_difference(set(all_imgs)))
                self.ids = annotated_imgs + unannotated_imgs[:len(annotated_imgs)]
            else:
                self.ids = get_anno_ids(annFile)
#         import pdb; pdb.set_trace()
        print(len(self.ids),'images will be used',flush=True)

        self.transform = transform
        self.bbox_transform = bbox_transform
        self.label_transform = label_transform
        # non flat list malignant lesions
#         self.list_malignant = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 29]
# non flat list all lesions
#         self.list_malignant = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
        # flat list all lesions (1 for mammo data, 2 for toy data
        self.list_malignant = [1,2]
        self.wrong_bbs = 0
        self.empty_annos = 0
        self.wrong_idx = 0
        self.size_mismatch = 0
        self.num_mal = 0
        self.num_ben = 0
        self.add_border = add_border

    def print_stats(self):
        print('So far {} malignant and {} benign lesions'.format(self.num_mal, self.num_ben), flush=True)
        print('There were {} wrong bbs'.format(self.wrong_bbs), flush=True)
        print('There were {} wrong idx'.format(self.wrong_idx), flush=True)
        print('There were {} empty annos'.format(self.empty_annos), flush=True)
        print('There were {} size mismatches'.format(self.size_mismatch), flush=True)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            sample: image, bounding box and label
        """
        coco = self.coco
        # Image id, e.g. 2
        try:
            img_id = self.ids[index]
        except IndexError:
            self.wrong_idx += 1
            # self.print_stats()
#             print('This is the {} th index error'.format(self.wrong_idx), flush=True)
            print('Index error',flush=True)
            return None
        # print('Img id',img_id,flush=True)

        # ID of annotation, TODO what does that mean?
        # e.g. [2]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # print('Ann ids',ann_ids,flush=True)

        # list with dict with area, bbox, etc.
        annotations = coco.loadAnns(ann_ids)
        # print('Annos',annotations,flush=True)
        
 
        if len(annotations) == 0:
            self.empty_annos += 1
        else:
            annotations = annotations[0]
#         print('Annos[0]',annotations,flush=True)

#         path = coco.loadImgs(img_id)[0]['file_name']
        try:
            path = coco.loadImgs(img_id)[0]['dataset_id'] # e.g."dp0200061903mr"
            side = get_side(path)
            path = path + '_downsampled.png'
        except KeyError:
            path = coco.loadImgs(img_id)[0]['file_name']
            side = get_side(path)
#         print('Path',path,flush=True)
        # print('Img data',coco.loadImgs(img_id)[0],flush=True)

        
#         print('Path',path,flush=True)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
#         print('Img shape 01',img.size,flush=True)
        # Convert to h,w,channels np array
        img = np.array(img)[:, :, :3]
#         print(img.min())
#         print(img.max())
        # print('Img shape 02',img.shape,flush=True)

        # Convert to channels,h,w np array
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)
        # print('Img shape 03',img.shape,flush=True)

        if img is None:
            print('Alaaaaarm, empty img', flush=True)
            print('Img id', img_id, flush=True)
            print('Annos', annotations, flush=True)
            return None
        
        if self.add_border:
            border_img = np.ones((img.shape[0],img.shape[1]+6,img.shape[2]+6))*(-1)
            border_img[:,3:-3,3:-3] = img
            img = np.copy(border_img)
            del border_img

        sample = {}
        sample['image'] = np.copy(img)
        sample['side'] = side

        # print('----',flush=True)
        # print('Size',sample['image'].shape,flush=True)
        # print('----',flush=True)
        # print('ID',annotations['image_id'],flush=True)
        # print('----',flush=True)
        # print('category_id',annotations['category_id'],flush=True)
        # print('----',flush=True)

        sample['original'] = np.copy(img)
        # sample['image_height'] = img.shape[1]
        # sample['image_width'] = img.shape[2]
        additive = 0
        if self.add_border:
            additive += 6
#         sample['image_height'] = coco.loadImgs(img_id)[0]['height'] + additive
#         sample['image_width'] = coco.loadImgs(img_id)[0]['width'] + additive
        
        sample['image_height'] = img.shape[1] + additive
        sample['image_width'] = img.shape[2] + additive
        
        if self.add_border:
            if not (img.shape[1] + additive == sample['image_height']) or not (img.shape[2] + additive == sample['image_width']):
                print('{} | {} | {} | {} | {}'.format(img_id,sample['image_height'],img.shape[1],sample['image_width'],img.shape[2]), flush=True)
                self.size_mismatch += 1
                return None
        else:
            if not (img.shape[1] == sample['image_height']) or not (img.shape[2] == sample['image_width']):
                print('{} | {} | {} | {} | {}'.format(img_id,sample['image_height'],img.shape[1],sample['image_width'],img.shape[2]), flush=True)
                self.size_mismatch += 1
#                 self.print_stats()
                print('This is the {} th mismatch'.format(self.size_mismatch), flush=True)
                # raise ValueError(
                #     'Not the correct img for this annotation:\n Annotation: h {}, w{}, Image: h {}, w{}'.format(
                #         sample['image_height'], sample['image_width'], img.shape[1], img.shape[2]))
                # print(
                #     'Not the correct img for this annotation:\n Annotation: h {}, w{}, Image: h {}, w{}'.format(
                #         sample['image_height'], sample['image_width'], img.shape[1], img.shape[2]),flush=True)
                #
                return None

#         if not (coco.loadImgs(img_id)[0]['id'] == annotations['image_id']):
#             raise ValueError('Not the same id, img {}, annotation {}'.format(coco.loadImgs(img_id)[0]['id'],
#                                                                              annotations['image_id']))

        
#         print('Category',annotations['category_id'],flush=True)
#         print('Is len annotations == 0?',len(annotations) == 0,flush=True)
#         print('Is category in list?',annotations['category_id'] in self.list_malignant,flush=True)
        if len(annotations) == 0:
            sample['image_id'] = img_id
            self.num_ben += 1
            sample['bbox'] = None
            sample['obbox'] = None
            sample['center_row'] = 0
            sample['center_col'] = 0
            sample['label'] = 0
            sample['has_lesion'] = 0
            sample_list = [sample]
        elif annotations['category_id'] in self.list_malignant:
            sample['image_id'] = annotations['image_id']
            self.num_mal += 1
            sample['obbox'] = annotations['bbox']
            sample['bbox'] = annotations['bbox']
            if (sample['image_height'] <= sample['obbox'][1]) or (sample['image_width'] <= sample['obbox'][0]):
                self.wrong_bbs += 1
                # self.print_stats()
                # print('This is the {} th weird bb'.format(self.wrong_bbs), flush=True)
                # raise ValueError('BBox out of image with height {} and width {} and bbox {}'.format(sample['image_height'],sample['image_width'],sample['obbox']))
                # print('BBox out of image with height {} and width {} and bbox {}'.format(sample['image_height'],sample['image_width'],sample['obbox']),flush=True)
                # print('Image id is',sample['image_id'],flush=True)
                # print('----',flush=True)
                return None
            sample['center_row'] = 112 / 224
            sample['center_col'] = 112 / 224
            sample['label'] = 1
            sample['has_lesion'] = 1
            sample_list = [sample]
            # print('Samples before bbox transform',flush=True)
            # print_without_img(sample_list)
        else:
            sample['image_id'] = img_id
            self.num_ben += 1
            sample['bbox'] = None
            sample['obbox'] = None
            sample['center_row'] = 0
            sample['center_col'] = 0
            sample['label'] = 0
            sample['has_lesion'] = 0
            sample_list = [sample]
#         print('Samples before bbox transform',flush=True)
#         print_without_img(sample_list)
        if self.bbox_transform is not None:
            # print(sample)
#             print('Before bbox transform',len(sample_list),flush=True)
            sample_list = self.bbox_transform(sample_list)
#             print('After bbox transform',len(sample_list),flush=True)
#         if len(sample_list) == 6:
#             print_without_img(sample_list)
#         print('Samples after bbox transform',flush=True)
        
        if self.label_transform is not None:
            sample_list = self.label_transform(sample_list)

        if self.transform is not None:
#             print('Before transform',len(sample_list),flush=True)
            sample_list = self.transform(sample_list)
#             print('After transform',len(sample_list),flush=True)
#         print('Samples after normal transform',flush=True)
#         print_without_img(sample_list)
        # self.print_stats()
        return sample_list

    def __len__(self):
        return len(self.ids)
