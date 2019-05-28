from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatino']
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def old_main(logdir, rundir):
#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_hits.csv'.format(logdir, rundir, rundir))))
    df = pd.read_csv('{}/{}/{}_hits.csv'.format(logdir, rundir, rundir))

    plt.figure()
    plt.plot(df['step'], df['total'])
    plt.plot(df['step'], df['correct'])

    plt.figure()
    acc = []
    for idx, dp in enumerate(df['total']):
        if dp == 0:
            acc.append(0)
        else:
            acc.append(df['correct'][idx] / dp)
    plt.plot(df['step'], acc)


def feat(logdir, rundir):
    which_stat = 'auc'
    legend_label = 'AUC'

#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}.csv'.format(logdir, rundir, which_stat))))
    df = pd.read_csv('{}/{}/{}.csv'.format(logdir, rundir, which_stat))

    plt.figure()
    plt.plot(df['step'], df['value'], color="#68BC36", label='Individual value')

    avg = 0
    running_avg = []
    for idx, dp in enumerate(df['value']):
        avg = (avg * idx + dp) / (idx + 1)
        running_avg.append(avg)

    plt.plot(running_avg, color='#F8696B', label='Running average')

    plt.gca().set_ylim(top=1)
    plt.xlabel('Epochs')
    plt.ylabel('{}'.format(legend_label))
    plt.title('Feature Network Validation Performance')
    plt.legend()


def feat_loss(logdir, rundir):
    which_stat = 'loss_val'
    legend_label = 'Loss'

#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}.csv'.format(logdir, rundir, which_stat))))
    df = pd.read_csv('{}/{}/{}.csv'.format(logdir, rundir, which_stat))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(df['step'], df['value'], color="#00B0FF", label='Training')

    avg = 0
    running_avg = []
    for idx, dp in enumerate(df['value']):
        avg = (avg * idx + dp) / (idx + 1)
        running_avg.append(avg)

    ax[0].plot(running_avg, color='#F8696B', label='Running average training')
    ax[0].set_ylim(top=1)
    ax[0].set_ylim(bottom=0)

    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('{}'.format(legend_label))
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()

    ax[0].legend()

    which_stat = 'loss_train'
    legend_label = 'Loss'

#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}.csv'.format(logdir, rundir, which_stat))))
    df_train = pd.read_csv('{}/{}/{}.csv'.format(logdir, rundir, which_stat))

    ax[1].plot(df_train['step'], df_train['value'], color="#68BC36", label='Validation')

    avg = 0
    running_avg = []
    for idx, dp in enumerate(df_train['value']):
        avg = (avg * idx + dp) / (idx + 1)
        running_avg.append(avg)

    ax[1].plot(running_avg, color='#F8696B', label='Running average validation')

    ax[1].set_ylim(top=1)
    ax[1].set_ylim(bottom=0)

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('{}'.format(legend_label))
    ax[1].legend()
    locations = np.linspace(0, len(df_train['step']), len(df['step']))
    steps = np.arange(0, len(df['step']), 10)
    ax[1].set_xticks(locations[steps])
    labels = np.arange(len(df['step']))[steps]
    ax[1].set_xticklabels(labels)

    plt.suptitle('Feature Network Loss')


def qnet_q_vals(logdir, rundir, runname, top_q=10, bottom_q=-10, top_trigger=20, bottom_trigger=-20):
    which_stat = 'loss_val'

    action_list = ['q_down',
                   'q_up',
                   'q_right',
                   'q_left',
                   'q_bigger',
                   'q_smaller',
                   'q_trigger',
                   ]

    legend_labels = ['Down', 'Up', 'Right', 'Left', 'Bigger', 'Smaller', 'Trigger']

    fig, ax = plt.subplots(4, 2)
    i = j = 0

    which_stat = action_list[0]
    legend_label = legend_labels[0]

#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))))
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))

    ax[i, j].plot(df['step'], df['value'], color="#00B0FF", label='Training')

    ax[i, j].set_ylim(top=top_q)
    ax[i, j].set_ylim(bottom=bottom_q)
    ax[i, j].set_ylabel('{}'.format(legend_label))

    ax[i, j].set_xlabel('Updates')
    ax[i, j].xaxis.set_label_position('top')
    ax[i, j].xaxis.tick_top()

    # ax[i,j].legend()

    j += 1
    if j > 1:
        j = 0
        i += 1

    for idx in range(1, 6):
        which_stat = action_list[idx]
        legend_label = legend_labels[idx]

#         print('Read data from {}'.format(os.path.abspath('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))))
        df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))

        ax[i, j].plot(df['step'], df['value'], color="#00B0FF", label='Training')
        ax[i, j].set_ylabel('{}'.format(legend_label))
        if i == 0 and j == 1:
            ax[i, j].set_xlabel('Updates')
            ax[i, j].xaxis.set_label_position('top')
            ax[i, j].xaxis.tick_top()
        elif i == 2 and j == 1:
            ax[i, j].set_xlabel('Updates')
        else:
            ax[i, j].set_xticks([])

        ax[i, j].set_ylim(top=top_q)
        ax[i, j].set_ylim(bottom=bottom_q)
        j += 1
        if j > 1:
            j = 0
            i += 1

    which_stat = action_list[6]
    legend_label = legend_labels[6]
#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))))
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))

    ax[i, j].plot(df['step'], df['value'], color="#00B0FF", label='Validation')

    ax[i, j].set_ylim(top=top_trigger)
    ax[i, j].set_ylim(bottom=bottom_trigger)

    ax[i, j].set_xlabel('Updates')
    ax[i, j].set_ylabel('{}'.format(legend_label))
    # ax[i,j].legend()
    plt.suptitle('Q-Values {}'.format(runname))
    ax[3, 1].axis('off')


def qnet_loss(logdir, rundir, max_val, runname):
    which_stat = 'loss'
    legend_label = 'Loss'

#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))))
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))

    fig, ax = plt.subplots(1, 1)
    ax.plot(df['step'], df['value'], color="#00B0FF", label='Training')

    avg = 0
    running_avg = []
    for idx, dp in enumerate(df['value']):
        avg = (avg * idx + dp) / (idx + 1)
        running_avg.append(avg)

    ax.plot(running_avg, color='#F8696B', label='Running average training')
    ax.set_ylim(top=max_val)
    ax.set_ylim(bottom=0)

    ax.set_xlabel('Updates')
    ax.set_ylabel('{}'.format(legend_label))
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.tick_top()

    ax.legend()

    plt.suptitle('Q-Net Loss {}'.format(runname))


def qnet_dist(logdir, rundir, runname):
    
    which_stat = 'hits'
    legend_label = 'Hits'
#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))))
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_distance.csv'.format(logdir, rundir, rundir))))
    df_eps = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'distance'))
    # steps = np.arange(0, len(df_eps), 10)
    # per_epoch_eps = df_eps['value'][steps]
    # print(len(per_epoch_eps))
    # print(len(df_eps))
    fig, ax = plt.subplots(2, 1)
#     print(df.head())
    total = ax[0].plot(df['step'], df['total'], color="#00B0FF", label='Total')

    correct = ax[0].plot(df['step'], df['correct'], color="#68BC36", label='Correct')
    ax2 = ax[0].twinx()
    # ax2.plot(range(len(per_epoch_eps)), per_epoch_eps, label='Epsilon')
    # ax2.plot(df_eps['step'], df_eps['min'], label='Min')
    dist = ax2.plot(df_eps['step'], df_eps['avg'], label='Average distance')
    ax2.set_ylabel('Distance\nground\ntruth & last\nbounding box')
    # ax[0].set_ylim(top=1)
    # ax[0].set_ylim(bottom=0)

    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Hits')
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()

    lns = total + correct + dist
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs)

    acc = []
    for idx, dp in enumerate(df['total']):
        if dp == 0:
            acc.append(0)
        else:
            acc.append(df['correct'][idx] / dp)

    acc = ax[1].plot(df['step'], acc, color="#68BC36",label='Accuracy')
    ax2 = ax[1].twinx()
    # ax2.plot(range(len(per_epoch_eps)), per_epoch_eps)
    # ax2.plot(df_eps['step'], df_eps['max'], label='Max')
    dist = ax2.plot(df_eps['step'], df_eps['avg'], label='Average distance')
    ax2.set_ylabel('Distance\nground\ntruth & last\nbounding box')

    # ax[1].set_ylim(top=1)
    # ax[1].set_ylim(bottom=0)

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    lns = acc + dist
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs)
    # ax[1].legend()

    # locations = np.linspace(0,len(df_eps['step']),len(df['step']))
    # steps = np.arange(0,len(df['step']),10)
    # ax[1].set_xticks(locations[steps])
    # labels = np.arange(len(df['step']))[steps]
    # ax[1].set_xticklabels(labels)

    plt.suptitle('Q-Net Training {}'.format(runname))


def qnet_hits_old(logdir, rundir,runname):
    which_stat = 'hits'
    legend_label = 'Hits'

#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))))
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_epsilon.csv'.format(logdir, rundir, rundir))))
    df_eps = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'epsilon'))
    # steps = np.arange(0, len(df_eps), 10)
    # per_epoch_eps = df_eps['value'][steps]
    # print(len(per_epoch_eps))
    # print(len(df_eps))

    fig, ax = plt.subplots(3, 1)
#     print(df.head())
    total = ax[0].plot(df['step'], df['total'], color="#00B0FF", label='Total')
    correct = ax[0].plot(df['step'], df['correct'], color="#68BC36", label='Correct')
    ax2 = ax[0].twinx()
    # ax2.plot(range(len(per_epoch_eps)), per_epoch_eps, label='Epsilon')
    # epsilon = ax2.plot(df_eps['step'], df_eps['value'], label='Epsilon')
    # ax2.set_ylabel('Epsilon')
    df_dist = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'distance'))
    dist = ax2.plot(df_dist['step'], df_dist['avg'], label='Average distance')
    
    ax2.set_ylabel('Distance\nground\ntruth & last\nbounding box')
    

    # ax[0].set_ylim(top=1)
    # ax[0].set_ylim(bottom=0)

    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Hits')
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()

    # added these three lines
    lns = total + correct + dist
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs)

    # ax[0].legend()

#     acc = []
#     for idx, dp in enumerate(df['total']):
#         if dp == 0:
#             acc.append(0)
#         else:
#             acc.append(df['correct'][idx] / dp)


    which_stat = 'lesions_acc'
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
    acc = ax[1].plot(df['step'],df['value'], color="#68BC36", label='Accuracy')
    ax2 = ax[1].twinx()
    # ax2.plot(range(len(per_epoch_eps)), per_epoch_eps)
    epsilon = ax2.plot(df_eps['step'], df_eps['value'], label='Epsilon')
    ax2.set_ylabel('Epsilon')

    ax[1].set_ylim(top=1.1)
    ax[1].set_ylim(bottom=-0.1)
    ax2.set_ylim(top=1.1)
    ax2.set_ylim(bottom=-0.1)

    ax[1].get_xaxis().set_visible(False)
    ax[1].set_ylabel('Accuracy')
    # ax[1].legend()

    # added these three lines
    lns = acc + epsilon
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs)

    # locations = np.linspace(0,len(df_eps['step']),len(df['step']))
    # steps = np.arange(0,len(df['step']),10)
    # ax[1].set_xticks(locations[steps])
    # labels = np.arange(len(df['step']))[steps]
    # ax[1].set_xticklabels(labels)

#     which_stat = 'per_img_hits'
    which_stat = 'score'
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
    per_img = ax[2].plot(df['step'], df['value'], color="#68BC36", label='Score')
    ax[2].plot(df['step'], np.zeros(len(df['step'])), color='#F8696B')
    ax2 = ax[2].twinx()
    # ax2.plot(range(len(per_epoch_eps)), per_epoch_eps)
    epsilon = ax2.plot(df_eps['step'], df_eps['value'], label='Epsilon')
    
    ax2.set_ylabel('Epsilon')

#     ax[2].set_ylim(top=1.1)
#     ax[2].set_ylim(bottom=-0.1)
    ax2.set_ylim(top=1.1)
    ax2.set_ylim(bottom=-0.1)

    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Score')
    # ax[1].legend()

    # added these three lines
    lns = per_img + epsilon
    labs = [l.get_label() for l in lns]
    ax[2].legend(lns, labs)

    plt.suptitle('Q-Net Training {}'.format(runname))

#####################################
#####################################
#####################################
#####################################


def qnet_hits(logdir, rdir,runname):
    which_stat = 'hits'
    legend_label = 'Hits'
    rundir = rdir

    fig, ax = plt.subplots(2, 1,figsize=(10,10))
   
    which_stat = 'lesions_acc'
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
    df_eps = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'epsilon'))
    ax[0].plot(df['step'][:63],df['value'][:63], color="#68BC36", label='_nolegend_')
    ax2 = ax[0].twinx()
    ax2.plot(df_eps['step'][:63], df_eps['value'][:63], color='#1f77b4', label='_nolegend_')
    end_value = df['value'][62]
    end_eps = df_eps['value'][62]

    rundir = 'resnet_final_35_1'
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
    df_eps = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'epsilon'))
    ax[0].plot([63,64],[end_value,df['value'][0]],color="#68BC36", label='_nolegend_')
    ax2.plot([63,64],[end_eps,df_eps['value'][0]], color='#1f77b4',label='_nolegend_')
    acc = ax[0].plot(df['step'],df['value'], color="#68BC36", label='Prec Lesion Lvl')
    
    # ax2.plot(range(len(per_epoch_eps)), per_epoch_eps)
    epsilon = ax2.plot(df_eps['step'], df_eps['value'], color='#1f77b4', label='Epsilon')
    ax2.set_ylabel('Epsilon')

    ax[0].set_ylim(top=1.1)
    ax[0].set_ylim(bottom=-0.1)
    ax2.set_ylim(top=1.1)
    ax2.set_ylim(bottom=-0.1)

    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Precision')
    # ax[1].legend()

    # added these three lines
    lns = acc + epsilon
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs)#,loc=5)

    # locations = np.linspace(0,len(df_eps['step']),len(df['step']))
    # steps = np.arange(0,len(df['step']),10)
    # ax[1].set_xticks(locations[steps])
    # labels = np.arange(len(df['step']))[steps]
    # ax[1].set_xticklabels(labels)

#     which_stat = 'per_img_hits'
    rundir = rdir
    which_stat = 'tp'
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
    which_stat = 'fn'
    df_fn = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
    df_eps = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'epsilon'))
    rec = []
    for idx in range(63):
        rec.append(df['value'][idx]/(df['value'][idx]+df_fn['value'][idx]))
    end_rec = df['value'][62]/(df['value'][62]+df_fn['value'][62])
    end_eps = df_eps['value'][62]

    per_img = ax[1].plot(df['step'][:63], rec, color="#68BC36", label='Rec Lesion Lvl')
    ax2 = ax[1].twinx()
    epsilon = ax2.plot(df_eps['step'][:63], df_eps['value'][:63],color='#1f77b4', label='Epsilon')
    

    ##########
    rundir = 'resnet_final_35_1'
    which_stat = 'tp'
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
    which_stat = 'fn'
    df_fn = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
    df_eps = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'epsilon'))
    rec = []
    for idx in range(len(df)):
        rec.append(df['value'][idx]/(df['value'][idx]+df_fn['value'][idx]))
    
    ax[1].plot([63,64],[end_rec,df['value'][0]/(df['value'][0]+df_fn['value'][0])],color="#68BC36", label='_nolegend_')
    ax2.plot([63,64],[end_eps,df_eps['value'][0]], color='#1f77b4',label='_nolegend_')
    ax[1].plot(df['step'], rec, color="#68BC36", label='_nolegend_')
    ax2.plot(df_eps['step'], df_eps['value'],color='#1f77b4', label='_nolegend_')
    ######


    ax2.set_ylabel('Epsilon')

    ax[1].set_ylim(top=1.1)
    ax[1].set_ylim(bottom=-0.1)
    ax2.set_ylim(top=1.1)
    ax2.set_ylim(bottom=-0.1)

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Recall')
    # ax[1].legend()

    # added these three lines
    lns = per_img + epsilon
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs)

    plt.suptitle('Q-Net Training {}'.format(runname))
    plt.show()

def qnet_dist_new(logdir, rdir,runname):
    which_stat = 'hits'
    legend_label = 'Hits'
    rundir = rdir
    fig, ax = plt.subplots(2, 1,figsize=(10,10))

#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))))
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_epsilon.csv'.format(logdir, rundir, rundir))))
    df_eps = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'epsilon'))


    ax[0].plot(df['step'][:63], df['total'][:63], color="#00B0FF", label='_nolegend_')
    ax[0].plot(df['step'][:63], df['correct'][:63], color="#68BC36", label='_nolegend_')
    end_total = df['total'][62]
    end_correct = df['correct'][62]


    rundir = 'resnet_final_35_1'
    df = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, which_stat))
#     print('Read data from {}'.format(os.path.abspath('{}/{}/{}_epsilon.csv'.format(logdir, rundir, rundir))))
    df_eps = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'epsilon'))
    # print(df.head())


    ax[0].plot([63,64],[end_total,df['total'][0]],color="#00B0FF", label='_nolegend_')
    ax[0].plot([63,64],[end_correct,df['correct'][0]],color="#68BC36", label='_nolegend_')

    total = ax[0].plot(df['step'], df['total'], color="#00B0FF", label='Total')
    correct = ax[0].plot(df['step'], df['correct'], color="#68BC36", label='Correct')

    # ax2 = ax[0].twinx()
    # ax2.plot(range(len(per_epoch_eps)), per_epoch_eps, label='Epsilon')
    # epsilon = ax2.plot(df_eps['step'], df_eps['value'], label='Epsilon')
    # ax2.set_ylabel('Epsilon')


    rundir = rdir
    df_dist = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'distance'))
    dist = ax[1].plot(df_dist['step'][:63], df_dist['avg'][:63], color="#1f77b4")
    end_avg = df_dist['avg'][62]
    rundir = 'resnet_final_35_1'

    df_dist = pd.read_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, 'distance'))
    ax[1].plot([63,64],[end_avg,df_dist['avg'][0]], color="#1f77b4")
    dist = ax[1].plot(df_dist['step'], df_dist['avg'], color="#1f77b4", label='Average distance')


    ax[1].set_ylabel('Distance\nground\ntruth & last\nbounding box')
    ax[0].legend()

    # ax[0].set_ylim(top=1)
    # ax[0].set_ylim(bottom=0)

    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('Hits')
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()
    plt.suptitle('Q-Net Training {}'.format(runname))
    plt.show()

def main(logdir, rundir, runname, which_plots=[True,True,True,True]):
    # main()
    # feat(logdir,rundir)
    # feat_loss(logdir,rundir)
    if which_plots[0]:
        qnet_q_vals(logdir, rundir, runname, 15, -1, 15, -20)
    if which_plots[1]:
        qnet_hits(logdir, rundir, runname)
    if which_plots[2]:
        qnet_loss(logdir, rundir, 0.2, runname)
    if which_plots[3]:
        qnet_dist(logdir, rundir, runname)

    plt.show()
    

if __name__ == '__main__':
    print('Hellohello')
    logdir = '../local_python'
    rundir = 'resnet_final_35'
    runname = 'FFDM, Simple CNN Backend'
    qnet_hits(logdir, rundir,runname)
    # do_qvals = False
    # do_hits = True
    # do_loss = False
    # do_dist = False
    # which_plots = [do_qvals, do_hits, do_loss, do_dist]
    # main(logdir, rundir , 'FFDM, Simple CNN Backend', which_plots)
