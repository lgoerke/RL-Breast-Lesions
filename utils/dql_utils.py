import numpy as np


###########################################################
# Training utils                                          #
###########################################################

def get_attribute_from_list(transition_list,attribute):
    new_list = []
    if attribute == 'next_state':
        for entry in transition_list:
            new_list.append(entry.next_state)
    elif attribute == 'state':
        for entry in transition_list:
            new_list.append(entry.state)
    elif attribute == 'action':
        for entry in transition_list:
            new_list.append(entry.action)
    elif attribute == 'reward':
        for entry in transition_list:
            new_list.append(entry.reward)
    elif attribute == 'resized_im':
        for entry in transition_list:
            new_list.append(entry.resized_im)
    return new_list


def transform_bb(a, row, col, row_no, col_no, img_r, img_c, factor_trans=None, factor_scale=None):
    if factor_trans:
        factor_translation = factor_trans
    else:
        factor_translation = 0.3
    if factor_scale:
        factor_scaling = factor_scale
    else:
        factor_scaling = 0.15
    # translate in row direction positively
    # DOWN
    if a == 0:
        new_row = int(row + (row_no * (factor_translation)))
        if (new_row + row_no) > img_r:
            new_row = img_r - row_no
        new_col = col
        new_row_no = row_no
        new_col_no = col_no
    # translate in row direction negatively
    # UP
    elif a == 1:
        new_row = int(row - (row_no * (factor_translation)))
        if new_row < 0:
            new_row = 0
        new_col = col
        new_row_no = row_no
        new_col_no = col_no
    # translate in col direction positively
    # RIGHT
    elif a == 2:
        new_col = int(col + (col_no * (factor_translation)))
        if (new_col + col_no) > img_c:
            new_col = img_c - col_no
        new_row = row
        new_row_no = row_no
        new_col_no = col_no
    # translate in col direction negatively
    # LEFT
    elif a == 3:
        new_col = int(col - (col_no * (factor_translation)))
        if new_col < 0:
            new_col = 0
        new_row = row
        new_row_no = row_no
        new_col_no = col_no
    # scale up action
    # BIGGER
    elif a == 4:
        if row_no >= img_r or col_no >= img_c:
            new_row = row
            new_col = col
            new_row_no = row_no
            new_col_no = col_no
        else:
            # print('Original',row,col,row_no,col_no)
            # scale by one sixth but keep bounding box at same position upper left corner
            new_row_no = int((1 + factor_scaling) * row_no)
            new_col_no = int((1 + factor_scaling) * col_no)
            # print('Growing',new_row_no,new_col_no)

            translation_rows = (new_row_no - row_no)//2
            translation_cols = (new_col_no - col_no)//2
            new_row = row - translation_rows
            new_col = col - translation_cols

            row_lim = img_r-new_row_no-1
            row_lim = np.clip(row_lim, 0, img_r)
            col_lim = img_c-new_col_no-1
            col_lim = np.clip(col_lim, 0, img_c)

            new_row = np.clip(new_row, 0, row_lim)
            new_col = np.clip(new_col, 0, col_lim)

            if ((new_row + new_row_no) > img_r):
                new_row_no = new_col_no = img_r - new_row
            if (new_col + new_col_no) > img_c:
                new_col_no = new_row_no = img_c - new_col


        # print('New',new_row,new_col,new_row_no,new_col_no)
    # scale down action
    # SMALLER
    elif a == 5:
        # scale by one sixth but keep bounding box at same position upper left corner
        new_row_no = int((1 - factor_scaling) * row_no)
        new_col_no = int((1 - factor_scaling) * col_no)

        if new_row_no < 1:
            new_row_no = new_col_no = 1
        if new_col_no < 1:
            new_col_no = new_ro_no = 1
        
        translation_rows = (row_no - new_row_no)//2
        translation_cols = (col_no - new_col_no)//2

        new_row = row + translation_rows
        new_col = col + translation_cols

        row_lim = img_r-new_row_no-1
        row_lim = np.clip(row_lim, 0, img_r)
        col_lim = img_c-new_col_no-1
        col_lim = np.clip(col_lim, 0, img_c)

        new_row = np.clip(new_row, 0, row_lim)
        new_col = np.clip(new_col, 0, col_lim)
    # trigger action
    # TRIGGER
    elif a == 6:
        new_col = col
        new_row = row
        new_row_no = row_no
        new_col_no = col_no
    return new_row, new_col, new_row_no, new_col_no


def string_case(c):
    s = None
    # Exploration rnd
    if c == 1:
        s = "explore rnd"
    # Guided rnd
    elif c == 2:
        s = "explore guide dist"
    # Guided pos
    elif c == 3:
        s = "explore guide pos"
    # Exploitation
    elif c == 4:
        s = "exploit"
    # Max steps
    elif c == 5:
        s = "max steps"
    else:
        s = 'none'
    return s


def string_action(a):
    s = None
    # DOWN
    if a == 0:
        s = "down"
    # UP
    elif a == 1:
        s = "up"
    # RIGHT
    elif a == 2:
        s = "right"
    # LEFT
    elif a == 3:
        s = "left"
    # BIGGER
    elif a == 4:
        s = "bigger"
    # SMALLER
    elif a == 5:
        s = "smaller"
    # TRIGGER
    elif a == 6:
        s = "trigger"
    else:
        s = 'none'
    return s
