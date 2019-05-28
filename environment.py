import numpy as np
import torch
from torch.utils.data import SequentialSampler, WeightedRandomSampler

import models as m
import random

from utils.datasets import mammo_collate
from utils.image_manipulation import resize as im_resize, resize_np
import utils.dql_utils as u
from utils.utils import calc_dice_np, get_mask_from_bb, calc_dice_np_specs, get_center_points
from utils.torch_utils import tensor_to_numpy, var_to_numpy, tensor_to_var, numpy_to_tensor
import os
from utils.misc_fcts_visualization import get_img_imshow
import matplotlib.pyplot as plt
from utils.utils import get_nums_from_bbox
import matplotlib.patches as patches
import imageio
import resnet as res
import scipy.spatial.distance as dist
from utils.dql_utils import string_action
from utils.transforms import check_bbox


class MammoEnv(object):

    def __init__(self, dataset, eta, zeta, tau, model, one_img=False, max_no_imgs=-1, seed=12345, with_hist=False,
                 iota=-1, dist_reward=False, dist_factor=1,f_one=False):
        """
        :param iota:
        :param dataloader (torch.utils.data.dataloader): Dataloader with images which are the environment. When
        resetting, next image is loaded
        :param eta (float): Value of positive and negative reward when trigger action is done
        :param zeta (float): Value of positive and negative reward when Dice score is bigger or smaller after action
        is done
        :param tau (float): Dice score threshold for positively labelled overlap
        :param model (nn.module): QNetwork
        :param one_img (boolean):
        :param max_no_imgs (int): how many images should be used from dataloader, if smaller than 1, all images are used
        :param seed (int): Seed for permutation of dataloader indices
        :param with_hist (boolean): If history should be attached to observations
        """
        self.one_img = one_img
        # self.one_img_idx = 932 # In the case that the whole dataset is used
        self.one_img_idx = 0
        self.dataiter = None
#         self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
#                                                       num_workers=os.cpu_count() - 1, collate_fn=mammo_collate)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                                                      num_workers=4, collate_fn=mammo_collate)
        # TODO dataloader
        self.f_one = f_one
        self.eta = eta
        self.zeta = zeta
        self.iota = iota
        self.tau = tau
        self.dist_reward = dist_reward
        self.dist_factor = dist_factor
        if isinstance(model, m.CombiNet) or isinstance(model, m.RCombiNet) or isinstance(model,
                                                                                         res.CombiNet) or isinstance(
            model, res.RCombiNet):
            self.combi = True
        else:
            self.combi = False
        print('Dataset has length', len(self.dataloader))
        np.random.seed(seed)
        random.seed(seed)

        if self.one_img:
            batch = next(iter(self.dataloader))
            self.orig_img = batch['image_batch'][0]
            self.gt_bbox = batch['bbox_batch'][0]
            self.gt_mask = get_mask_from_bb(self.orig_img, self.gt_bbox)
            self.img_size_rows = batch['image_height_batch'][0]
            self.img_size_cols = batch['image_width_batch'][0]
            self.breast_side = batch['side_batch'][0]
            smaller_side = min(self.img_size_rows, self.img_size_cols)
            self.init_bb_size = int(smaller_side * 0.75)
            self.init_bb_row = int((self.img_size_rows - self.init_bb_size) // 2)
            self.init_bb_col = int((self.img_size_cols - self.init_bb_size) // 2)
        else:
            # Get list of malignant lesion imgs
            if self.f_one:
                mal_ims = 0
                other_ims = 0
                weights = []
                img_limit = max_no_imgs if max_no_imgs > 0 else np.inf
                for idx, batch in enumerate(self.dataloader, 1):
                    if mal_ims + other_ims >= img_limit*2:
                        weights.append(0)
                    else:
                        if len(batch['label_batch']) > 0:
                            if batch['label_batch'][0] == 1:
                                if mal_ims < img_limit:
                                    weights.append(1)
                                    mal_ims += 1
                                else:
                                    weights.append(0)
                            else:
                                if other_ims < img_limit:
                                    weights.append(1)
                                    other_ims += 1
                                else:
                                    weights.append(0)
                        else:
                            weights.append(0)
                weights = np.array(weights)
                num_mal = int(np.sum(weights))
                print('Found {} images which will be sampled'.format(num_mal), flush=True)

                self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                              sampler=WeightedRandomSampler(weights=
                                                                  numpy_to_tensor(weights).double(), num_samples=num_mal,
                                                                  replacement=False),
                                                              num_workers=4, collate_fn=mammo_collate)
            else:
                is_malignant = []
                img_limit = max_no_imgs if max_no_imgs > 0 else np.inf
                for idx, batch in enumerate(self.dataloader, 1):
                    if np.sum(np.array(is_malignant)) >= img_limit:
                        is_malignant.append(0)
                    else:
                        if len(batch['label_batch']) > 0:
                            if batch['label_batch'][0] == 1:
                                is_malignant.append(1)
                            else:
                                is_malignant.append(0)
                        else:
                            is_malignant.append(0)
                is_malignant = np.array(is_malignant)
                num_mal = int(np.sum(is_malignant))
                print('Found {} images with malignant lesion'.format(num_mal), flush=True)

                self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                              sampler=WeightedRandomSampler(weights=
                                                                  numpy_to_tensor(is_malignant).double(), num_samples=num_mal,
                                                                  replacement=False),
                                                              num_workers=4, collate_fn=mammo_collate)
            self.no_imgs = num_mal
            # images, labels = self.dataiter.next()



            self.img_idx = -1

            self.orig_img = None
            self.gt_bbox = None
            self.img_size_rows = None
            self.img_size_cols = None
            self.init_bb_size = None
            self.init_bb_row = None
            self.init_bb_col = None
            self.breast_side = None

        self.feature_size = 224
        self.action_history = np.zeros((70,))
        self.action_history_base_ten = []
        self.with_hist = with_hist

    def __len__(self):
        """
        :return: How many images the environment goes through
        """
        return self.no_imgs

    def reset(self, fcn=False):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        images_left = True
        if not self.one_img:
            # start_time = start_measuring_time()
            self.img_idx += 1
            if self.img_idx == self.no_imgs:
                self.img_idx = 0
            elif self.img_idx == (self.no_imgs - 1):
                images_left = False

            if self.img_idx == 0:
                del self.dataiter
                self.dataiter = iter(self.dataloader)

            try:
                batch = next(self.dataiter)
            except StopIteration:
                raise (StopIteration, 'Stop, idx {}, no imgs {}'.format(self.img_idx, self.no_imgs))

#             print('Loaded image {}'.format(self.img_idx))
#             print('Image id {}'.format(batch['image_id_batch'][0]))
#             print('Images left is {}'.format(images_left))
            self.orig_img = batch['image_batch'][0]
#             print('Correct shape?', self.orig_img.shape)
            self.img_size_rows = batch['image_height_batch'][0]
            self.img_size_cols = batch['image_width_batch'][0]
            self.breast_side = batch['side_batch'][0]
            smaller_side = min(self.img_size_rows, self.img_size_cols)
            self.init_bb_size = int(smaller_side * 0.75)
            self.init_bb_row = int((self.img_size_rows - self.init_bb_size) // 2)
            self.init_bb_col = int((self.img_size_cols - self.init_bb_size) // 2)
            
            self.gt_bbox = batch['bbox_batch'][0]
            

            # stop_measuring_time(start_time, "Get from dataset")

            if self.f_one:
                sampled_bb = None
                self.gt_mask = None
            else:
                self.gt_mask = get_mask_from_bb(self.orig_img, self.gt_bbox)
                # start_time = start_measuring_time()
                sampled_bb = self.sample_starting_bb()
                # stop_measuring_time(start_time, "Sample starting bb")
                print('01 Sample bb', sampled_bb, 'rows', self.img_size_rows, 'cols', self.img_size_cols, flush=True)

            if sampled_bb:
                self.init_bb_size = sampled_bb[2]
                self.init_bb_row = sampled_bb[0]
                self.init_bb_col = sampled_bb[1]
            else:
                # Default bb spans 75% of smaller side of image
                smaller_side = min(self.img_size_rows, self.img_size_cols)
                self.init_bb_size = int(smaller_side * 0.75)
                self.init_bb_row = int((self.img_size_rows - self.init_bb_size) // 2)
                self.init_bb_col = int((self.img_size_cols - self.init_bb_size) // 2)

        else:
            images_left = False
            sampled_bb = self.sample_starting_bb()
            print('02 Sample bb', sampled_bb,  'rows', self.img_size_rows, 'cols', self.img_size_cols, flush=True)
            
            if sampled_bb:
                self.init_bb_size = sampled_bb[2]
                self.init_bb_row = sampled_bb[0]
                self.init_bb_col = sampled_bb[1]
            else:
                # Default bb spans 75% of smaller side of image
                smaller_side = min(self.img_size_rows, self.img_size_cols)
                self.init_bb_size = int(smaller_side * 0.75)
                self.init_bb_row = int((self.img_size_rows - self.init_bb_size) // 2)
                self.init_bb_col = int((self.img_size_cols - self.init_bb_size) // 2)

        self.row = self.init_bb_row
        self.col = self.init_bb_col
        self.row_no = self.col_no = self.init_bb_size
        self.action_history_base_ten = []
        # start_time = start_measuring_time()
        resized_im = im_resize(self.orig_img, self.row, self.col, self.row_no, self.col_no, self.feature_size,
                               divide255=True)
        # start_time = start_measuring_time()
        see_lesion = True if self.get_current_dice() > 0 else False
        # stop_measuring_time(start_time, "Is see lesion in reset")
        return tensor_to_numpy(resized_im), images_left, see_lesion, self.action_history

    def step(self, action, fcn=False, sub_tau=0):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, has_moved).
        Args:
            action (object): an action provided by the environment
            fcn (boolean): if feature network is fully convolutional
            sub_tau (float): how much should be subtracted from target tau
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step()
                            calls will return undefined results
            has_moved (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # start_time = start_measuring_time()
        self.action_history_base_ten.append(action)
        new_row, new_col, new_row_no, new_col_no = u.transform_bb(action, self.row, self.col, self.row_no, self.col_no,
                                                                  self.img_size_rows, self.img_size_cols)
        # stop_measuring_time(start_time, "Transform_bb in step")

        # start_time = start_measuring_time()
        see_lesion = True if self.get_current_dice_from_bb(new_row, new_col, new_row_no, new_col_no) > 0 else False
        # stop_measuring_time(start_time, 'Is see lesion in step')

        # start_time = start_measuring_time()
        reward, done, has_moved, _ = self.calculate_reward(new_row, new_col, new_row_no, new_col_no, action, sub_tau,
                                                           see_lesion)
        # stop_measuring_time(start_time, 'Calculate_reward in step')

        self.row = new_row
        self.col = new_col
        self.row_no = new_row_no
        self.col_no = new_col_no

        self.add_new_action_to_hist(action)
        # start_time = start_measuring_time()
        try:
            # start_time = start_measuring_time()
            resized_im = im_resize(self.orig_img, self.row, self.col, self.row_no, self.col_no, self.feature_size,
                                   divide255=True)
            # stop_measuring_time(start_time, 'Resize in step')
            # resized_im_np = resize_np(self.orig_img, self.row, self.col, self.row_no, self.col_no, self.feature_size)
            # print('TORCH', resized_im.shape)
            # print('TORCH', resized_im[0, 0, 220:, :5])
            # print('TORCH', resized_im[0, 1, 220:, :5])
            # print('TORCH', resized_im[0, 2, 220:, :5])
            # print('NP', resized_im_np.shape)
            # print('NP', resized_im_np[0, 0, 220:, :5])
            # print('NP', resized_im_np[0, 2, 220:, :5])
            # print('NP', resized_im_np[0, 1, 220:, :5])
            # print(np.abs(resized_im[0, :, 220:, :5] - resized_im_np[0, :, 220:, :5]))
            # start_time = start_measuring_time()
            # resized_im = resized_im.unsqueeze(0)
            # stop_measuring_time(start_time, 'Unsqueeze in step')
        except AttributeError as err:
            print('Step with action', action)
            print('Orig img shape', self.orig_img.shape)
            print('Height', self.img_size_rows)
            print('Width', self.img_size_cols)
            print('Row', self.row)
            print('Col', self.col)
            print('Row no', self.row_no)
            print('Col no', self.col_no)
            print('Feature size', self.feature_size)
            raise AttributeError(err)
        # stop_measuring_time(start_time, 'Resize and unsqueeze in step')

        return tensor_to_numpy(resized_im), reward, done, has_moved, see_lesion, self.action_history

    def reset_with_given_bb(self, r, c, rn, cn, fcn=False):
        """Resets the state of the environment with a given bb on the current image
        Returns: observation (object): the initial observation of the
            space.
        """
        self.row = r
        self.col = c
        self.row_no = rn
        self.col_no = cn

        resized_im = im_resize(self.orig_img, self.row, self.col, self.row_no, self.col_no, self.feature_size,
                               divide255=True)
        if resized_im is None:
            return None, None, None
        see_lesion = True if self.get_current_dice() > 0 else False
        return tensor_to_numpy(resized_im), see_lesion, self.action_history

    def get_bbs_testing(self,test_mode=3):
        """
        Get 7 testing bbs. One in center with 75% size (smaller side), 6 with 50% size (smaller side) in top left,
        top right, center left, center right, bottom left, bottom right
        :return: list of row values, column values, bbs heights and bbs widths
        """
        if test_mode == 0 or test_mode > 1:
        
            cols075 = int(self.img_size_cols * 0.75)
            bb_rows = [(self.img_size_rows - cols075) // 2]
            bb_cols = [(self.img_size_cols - cols075) // 2]
            bb_row_nos = [cols075]
            bb_col_nos = [cols075]
        
        if test_mode > 0:
        
            if (self.img_size_rows / self.img_size_cols) <= 3 / 2:
                # overlap rows
                cols050 = int(self.img_size_cols * 0.50)
                smaller_size = cols050
                overlap = (3 * cols050 - self.img_size_rows) // 2
                bb_rows.extend([0, 0, cols050 - overlap, cols050 - overlap,
                           int(self.img_size_rows - cols050), int(self.img_size_rows - cols050)])

            else:
                # overlap cols
                rows033 = int(self.img_size_rows * (1 / 3))
                smaller_size = rows033
                bb_rows.extend([0, 0, rows033, rows033,
                           int(self.img_size_rows - rows033), int(self.img_size_rows - rows033)])

            bb_cols.extend([0, int(self.img_size_cols - smaller_size), 0,
                       int(self.img_size_cols - smaller_size), 0, int(self.img_size_cols - smaller_size)])

            bb_row_nos.extend([smaller_size, smaller_size, smaller_size, smaller_size, smaller_size, smaller_size])
            bb_col_nos.extend([cols075, smaller_size, smaller_size, smaller_size, smaller_size, smaller_size, smaller_size])

        return bb_rows, bb_cols, bb_row_nos, bb_col_nos

    
    
    def get_action_smaller_dist(self):
        min_dist = np.inf
        min_action = -1
        for action in range(7):
            new_row, new_col, new_row_no, new_col_no = u.transform_bb(action, self.row, self.col, self.row_no,
                                                                      self.col_no, self.img_size_rows,
                                                                      self.img_size_cols)
            new_dist = self.get_current_distance_center_from_bb(new_row, new_col, new_row_no, new_col_no,'euclidean')
            if new_dist < min_dist:
                min_dist = new_dist
                min_action = action
        return min_action
            
    
    def get_positive_actions(self, sub_tau=0):
        """

        :param sub_tau: how much should be subtracted from target tau
        :return: action list, all rewards list, positive rewards list, whether there is positive trigger possible
        """

        rewards = np.zeros((7, 1))
        dice_scores = np.zeros((7, 1))
        got_eta = False
        for i, action in enumerate(range(7)):
            new_row, new_col, new_row_no, new_col_no = u.transform_bb(action, self.row, self.col, self.row_no,
                                                                      self.col_no, self.img_size_rows,
                                                                      self.img_size_cols)
            see_lesion = True if self.get_current_dice_from_bb(new_row, new_col, new_row_no, new_col_no) > 0 else False
            rewards[i], _, _, dice_scores[i] = self.calculate_reward(new_row, new_col, new_row_no, new_col_no, action,
                                                                     sub_tau, see_lesion)

        actions = []
        pos_rewards = []
        best_dice = np.max(dice_scores)
        for a, r in enumerate(rewards):
            if r > 0:
                # This line makes agent only choose best possible action, otherwise one of positive actions
                if dice_scores[a] == best_dice:
                    actions.append(a)
                    pos_rewards.append(r)
            if r == self.eta:
                got_eta = True
        return actions, rewards, pos_rewards, got_eta

    def get_current_dice_from_bb(self, row, col, row_no, col_no):

        # mask_of_search_bb = get_mask_from_bb(self.orig_img, [col, row, col_no, row_no])
        # return calc_dice_np(tensor_to_numpy(mask_of_search_bb), tensor_to_numpy(self.gt_mask))
        if self.get_original_bb() is None:
            return 0
        return calc_dice_np_specs([col, row, col_no, row_no], self.get_original_bb())
    
    def get_current_distance_center_from_bb(self, row, col, row_no, col_no,measure='chebyshev'):
        if self.get_original_bb() is None:
            return 0
        center_current = get_center_points([col, row, col_no, row_no])
        center_original = get_center_points(self.get_original_bb())
        return dist.cdist(center_current, center_original, measure)[0, 0]

    def get_current_distance_center(self,measure='chebyshev'):
        if self.get_original_bb() is None:
            return np.inf
        center_current = get_center_points(self.get_current_bb())
        center_original = get_center_points(self.get_original_bb())
        return dist.cdist(center_current, center_original, measure)[0, 0]

    def get_current_dice(self):
        """Gets the dice score between the current bounding box and image
        Returns:
            dice: dice score
        """
        # mask_of_lesion = self.current_mask[self.row:self.row + self.row_no, self.col:self.col + self.col_no]
        # bb = torch.ones((self.row_no, self.col_no))

        # mask_of_search_bb = get_mask_from_bb(self.orig_img, self.get_current_bb())
        #
        # return calc_dice_np(tensor_to_numpy(mask_of_search_bb), tensor_to_numpy(self.gt_mask))
        if self.get_original_bb() is None:
            return 0
        return calc_dice_np_specs(self.get_current_bb(), self.get_original_bb())

    def get_bbs_plotting(self):
        """
        Get bb starting points for different resolutions
        :return: list of row values, column values, bbs heights and bbs widths
        """
        # 5 tiles
        cols020 = int(self.img_size_cols * 0.2)
        # 10 tiles
        cols010 = int(self.img_size_cols * 0.1)
        # 25 tiles
        cols004 = int(self.img_size_cols * 0.04)
        # 50 tiles
        # cols002 = int(self.img_size_cols*0.02)
        # 100 tiles
        # cols001 = int(self.img_size_cols*0.01)

        bb_rows = []
        bb_row_nos = []
        bb_col_nos = []
        bb_cols = []

        # for i,divisor in enumerate([cols020,cols010,cols004,cols002,cols001]):
        for i, divisor in enumerate([cols020, cols010, cols004]):
            _bb_rows = []
            _bb_row_nos = []
            _bb_col_nos = []
            _bb_cols = []

            translation_stride = int(divisor * 0.3)
            for j in range(self.img_size_rows // translation_stride):
                _bb_row_nos.append(divisor)
                _bb_col_nos.append(divisor)
                _bb_cols.append(0)
                _bb_rows.append(j * translation_stride)

            bb_row_nos.append(_bb_row_nos)
            bb_rows.append(_bb_rows)
            bb_col_nos.append(_bb_col_nos)
            bb_cols.append(_bb_cols)

        return bb_rows, bb_cols, bb_row_nos, bb_col_nos

    def render(self, mode='human', close=False, with_axis=True, with_state=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - figure: reutrns matplotlib figure
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.nxfdarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == 'human':
            filename = 'dummy.png'
            cm = 'binary'
            lwidth = 2

            image = self.get_original_img()
            r, c, rn, cn = get_nums_from_bbox(self.get_current_bb())
            # print(r, c, rn, cn)

            ro, co, rno, cno = get_nums_from_bbox(self.get_original_bb())
            # print(ro, co, rno, cno)

            if with_state:
                fig, ax = plt.subplots(1, 2)
            else:
                fig, ax = plt.subplots(1, 1)
            if not with_axis:
                if with_state:
                    ax[0].set_axis_off()
                else:
                    ax.set_axis_off()

            # Create a Rectangle patch
            edgecol = 'g'
            rect_gt = patches.Rectangle((co, ro), cno, rno, linewidth=lwidth, edgecolor=edgecol, facecolor='none')
            # Add the patch to the Axes

            edgecol = 'r'
            rect_current = patches.Rectangle((c, r), cn, rn, linewidth=lwidth, edgecolor=edgecol, facecolor='none')
            # Add the patch to the Axes
            if with_state:
                # print('MIN im', np.min(image))
                # print('MAX im', np.max(image))
                # print('MEAN im', np.mean(image))
                ax[0].imshow(get_img_imshow(image), cmap=cm)
                ax[0].add_patch(rect_gt)
                ax[0].add_patch(rect_current)
            else:
                ax.imshow(get_img_imshow(image), cmap=cm)
                ax.add_patch(rect_gt)
                ax.add_patch(rect_current)

            if with_state:
                ax[1].set_axis_off()
                # print('MIN', torch.min(self.get_current_state()))
                # print('MAX', torch.max(self.get_current_state()))
                # print('MEAN', torch.mean(self.get_current_state()))
                ax[1].imshow(get_img_imshow(self.get_current_state()), cmap=cm)

            plt.savefig(filename, bbox_inches='tight')
            img = imageio.imread(filename)
            # print(img.shape)
            os.remove(filename)
            return img
        elif mode == 'figure':
            cm = 'binary'
            lwidth = 2

            image = self.get_original_img()
            r, c, rn, cn = get_nums_from_bbox(self.get_current_bb())
            # print(r, c, rn, cn)

            ro, co, rno, cno = get_nums_from_bbox(self.get_original_bb())
            # print(ro, co, rno, cno)

            fig, ax = plt.subplots(1, 1)
            if not with_axis:
                ax.set_axis_off()
            ax.imshow(get_img_imshow(image), cmap=cm)
            # Create a Rectangle patch
            edgecol = 'g'
            rect = patches.Rectangle((co, ro), cno, rno, linewidth=lwidth, edgecolor=edgecol, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

            edgecol = 'r'
            rect = patches.Rectangle((c, r), cn, rn, linewidth=lwidth, edgecolor=edgecol, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            return fig
        else:
            raise NotImplementedError

    def is_hit(self, sub_tau=0):
        """
        :param sub_tau:
        :return: Whether current bb overlaps more than tau - sub_tau (Dice score)
        """
        return self.get_current_dice() > (self.tau - sub_tau)


    def is_one_img(self):
        return self.one_img

    def sample_starting_size(self, r, c):
        """
        Sample bb size according to upper left corner spec
        :param r: row value of upper left corner
        :param c: column value of upper left corner
        :return: random size of bb
        """
        size = 224
        # print('Row', r,flush=True)
        # print('Col', c,flush=True)
        # print('Img rows', self.img_size_rows,flush=True)
        # print('Img cols', self.img_size_cols,flush=True)
        # Check if there is enough space for the bb
        if self.img_size_cols - c < 224:
            size = self.img_size_cols - c - 1
            # print('Size change -01',size,flush=True)
        if self.img_size_rows - r < size:
            size = self.img_size_rows - r - 1
            # print('Size change -02', size,flush=True)
        # If no space restrictions where found, sample size
        if size == 224:
            if self.img_size_cols - c < self.img_size_rows - r:
                smaller_side = self.img_size_cols - c - 1
                # print('Smaller side -01', smaller_side,flush=True)
            else:
                smaller_side = self.img_size_rows - r - 1
                # print('Smaller side -02', smaller_side,flush=True)
            size = int(np.random.uniform(low=224, high=smaller_side))
            # print('Size sampled',size,flush=True)
        if size == 0:
            raise ValueError('Row {} and col {} lead to size {}'.format(r, c, size))
        return size

    def check_distance_edges(self, r, c, dist_edge):
        if self.img_size_rows - r < dist_edge:
            r = self.img_size_rows - dist_edge
        if self.img_size_cols - c < dist_edge:
            c = self.img_size_cols - dist_edge
        return r, c
    
    def sample_starting_bb(self):
        """
        :return: tuple with row value, column value and size of bb
        """
        thresh = 0.5
        dist_edge = 200
        if np.random.rand() < thresh: # close to ground truth
            ratio_r = np.random.uniform(0, 1)
            ratio_c = np.random.uniform(0,1)
            lim_r = min((self.gt_bbox[1]-1)//ratio_r,(self.img_size_rows-self.gt_bbox[1])//(1-ratio_r))
            lim_c = min((self.gt_bbox[0]-1)//ratio_c,(self.img_size_cols-self.gt_bbox[0])//(1-ratio_c))
            lim_all = min(lim_r,lim_c)
            if lim_all <= dist_edge:
                size = lim_all
            else:
                size = np.random.uniform(dist_edge, lim_all)
            row = self.gt_bbox[1] - size*ratio_r
            col = self.gt_bbox[0] - size*ratio_c
            row, col, size, _ = check_bbox(row, col, size, size, self.img_size_rows, self.img_size_cols)
            if size <= 50:
                return None
            return int(row), int(col), int(size)
        else: #Uniform over whole image
            c = np.random.uniform(low=1,high=self.img_size_cols-dist_edge-1)
            r = np.random.uniform(low=1, high=self.img_size_rows-dist_edge-1)
            r, c = self.check_distance_edges(r, c, dist_edge)
            # if distance to bottom smaller than distance to right border
            if self.img_size_rows - r < self.img_size_cols - c:
                if dist_edge >= self.img_size_rows - r:
                    size = self.img_size_rows - r - 1
                else:
                    size = np.random.uniform(low=dist_edge, high=self.img_size_rows -r - 1)
            else:
                if dist_edge >= self.img_size_cols - c:
                    size = self.img_size_cols - c - 1
                else:
                    size = np.random.uniform(low=dist_edge, high=self.img_size_cols - c - 1)
            r, c, size, _ = check_bbox(r, c, size, size, self.img_size_rows, self.img_size_cols)
            if size <= 50:
                return None
            return int(r), int(c), int(size)

    def get_tau(self):
        """
        :return: Dice score threshold value
        """
        return self.tau

    def get_original_bb(self):
        """
        :return: bb around tumor
        """
        return self.gt_bbox

    def get_original_img(self):
        """
        :return: Original mammography image (full size)
        """
        return self.orig_img

    def get_current_state(self):
        return torch.squeeze(
            im_resize(self.orig_img, self.row, self.col, self.row_no, self.col_no, self.feature_size, divide255=True))

    def get_current_bb(self):
        """
        :return: current bb of agent in bbox format [x (col), y (row), width (col_no), height (row_no)]
        """
        return self.col, self.row, self.col_no, self.row_no

    def get_current_bb_list(self):
        """
        :return: current bb of agent in list
        """
        return [self.row, self.col, self.row_no, self.col_no]

    def has_moved(self, new_row, new_col, new_row_no, new_col_no):
        """
        Checks if bb has moved
        :param new_row:
        :param new_col:
        :param new_row_no:
        :param new_col_no:
        :return: boolean
        """
        if new_row == self.row and new_col == self.col and new_row_no == self.row_no and new_col_no == self.col_no:
            return False
        else:
            return True

    def get_string_action_combi(self, num1, num2):
        action1 = string_action(num1)
        action2 = string_action(num2)
        if (action1 == 'left' and action2 == 'right') or (action2 == 'left' and action1 == 'right'):
            return 'lr'
        elif (action1 == 'up' and action2 == 'down') or (action1 == 'down' and action2 == 'up'):
            return 'ud'
        elif (action1 == 'smaller' and action2 == 'bigger') or (action1 == 'bigger' and action2 == 'smaller'):
            return 'sb'
        elif (action1 in ['left', 'right', 'up', 'down'] and action2 in ['smaller', 'bigger']) or (
                action2 in ['left', 'right', 'up', 'down'] and action1 in ['smaller', 'bigger']):
            return 'tz'
        elif (action1 in ['left', 'right'] and action2 in ['up', 'down']) or (
                action2 in ['left', 'right'] and action1 in ['up', 'down']):
            return 'tt'
        else:
            return action1 + action2

    def is_oscillating(self, limit):
        if len(self.action_history_base_ten) >= limit:
            one_back = self.action_history_base_ten[-1]
            two_back = self.action_history_base_ten[-2]
            if one_back == two_back:
                return False, None
            osc_cnt = 0
            cnt = 0
            for i in np.arange(-3, -limit - 1, -2):
                cnt += 1
                if self.action_history_base_ten[i] == one_back and self.action_history_base_ten[i - 1] == two_back:
                    osc_cnt += 1
            return cnt == osc_cnt, self.get_string_action_combi(one_back, two_back)
        else:
            return False, None

    def calculate_reward(self, new_row, new_col, new_row_no, new_col_no, action, sub_tau, see_lesion=False):
        """
        Calculate reward according to new bb. If action was trigger, larger reward is returned
        :param new_row:
        :param new_col:
        :param new_row_no:
        :param new_col_no:
        :param action:
        :param sub_tau: will b subtracted from Dice score threshold
        :return: reward, bool whether trigger action was taken and bool whether bb has moved
        """
        has_moved = self.has_moved(new_row, new_col, new_row_no, new_col_no)
        new_dice = self.get_current_dice_from_bb(new_row, new_col, new_row_no, new_col_no)
        new_dist = self.get_current_distance_center_from_bb(new_row, new_col, new_row_no, new_col_no)
        done = False
        if action == 6:
            done = True
            if sub_tau:
                if new_dice >= (self.tau - sub_tau):
                    reward = self.eta
                elif see_lesion:
                    if self.iota > -1:
                        reward = -self.iota
                    else:
                        reward = -self.eta
                else:
                    reward = -self.eta
            else:
                if new_dice >= self.tau:
                    reward = self.eta
                elif see_lesion:
                    if self.iota > -1:
                        reward = -self.iota
                    else:
                        reward = -self.eta
                else:
                    reward = -self.eta
        else:
            old_dist = self.get_current_distance_center()
            old_dice = self.get_current_dice()
            reward = 0
            if (new_dice - old_dice) < 0:
                reward += -self.zeta
            elif (new_dice - old_dice) > 0:
                reward += self.zeta
            if self.dist_reward:
                # only give distance reward when lesion is in bb
                if old_dice > 0 and new_dice > 0:
                    r = min(1, (4 / self.row_no) * old_dist) * self.zeta
                    if (new_dist - old_dist) > 0:
                        reward += -r * self.dist_factor
                    elif (new_dist - old_dist) < 0:
                        reward += r * self.dist_factor

        return reward, done, has_moved, new_dice

    def add_new_action_to_hist(self, action):
        """
        Append new one hot encoding of current action to history
        :param action:
        """
        a_vec = np.zeros((7,))
        a_vec[action] = 1
        self.action_history = self.action_history[7:]
        self.action_history = np.concatenate((self.action_history, a_vec))