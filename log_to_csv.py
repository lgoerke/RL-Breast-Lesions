import pandas as pd
# from tensorflow.python.summary import EventAccumulator
import tensorboard.backend.event_processing.event_accumulator as evacc
import os
from os import listdir
from os.path import isfile, join


def feat(logdir, rundir):
    more_dir = 'qnet/qnet/val/hits/total'
    # eventsname = 'events.out.tfevents.1532184516.d66112711c6d'

    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]

    # TODO
    file = os.path.join(logdir, rundir, more_dir, eventsname)

    file = os.path.join(logdir, rundir, eventsname)

    print('Creating accumulator from {}'.format(os.path.abspath(file)))

    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })

    ea.Reload()  # loads events from file

#     for key, entry in ea.Tags().items():
#         print(key, entry)

    df = pd.DataFrame(ea.Scalars('resnet/val/auc'))

#     print(df.head())
#     print(list(df))
    df = df.drop(columns=['wall_time'])
#     print(df.head())
#     print(list(df))

    file = os.path.join(logdir, rundir, 'auc.csv')
    df.to_csv(file)


def qnet_eps(logdir, rundir):
    more_dir = 'qnet'
    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
#     print(onlyfiles)
    eventsname = onlyfiles[0]
#     print(eventsname)

    file = os.path.join(logdir, rundir, more_dir, eventsname)

    print('Creating accumulator from {}'.format(os.path.abspath(file)))
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })

#     print('Loading events')
    ea.Reload()  # loads events from file

#     print(ea.Tags())

    df = pd.DataFrame(ea.Scalars('qnet/train/epsilon'))
    df = df.drop(columns=['wall_time'])
#     print(df.head())

    df.to_csv('{}/{}/{}_epsilon.csv'.format(logdir, rundir, rundir))


def qnet_q_vals(logdir, rundir):
    more_dir = 'qnet/qnet/train_stats'

    action_list = ['q_bigger',
                   'q_left',
                   'q_max',
                   'q_right',
                   'q_smaller',
                   'q_trigger',
                   'q_down',
                   'q_up']

    for action in action_list:
        onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir, action, 'value')) if
                     os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir, action, 'value'), f))]
        eventsname = onlyfiles[0]

        file = os.path.join(logdir, rundir, more_dir, action, 'value', eventsname)

        print('Creating accumulator from {}'.format(os.path.abspath(file)))
        ea = evacc.EventAccumulator(file,
                                                size_guidance={  # see below regarding this argument
                                                    evacc.COMPRESSED_HISTOGRAMS: 500,
                                                    evacc.IMAGES: 4,
                                                    evacc.AUDIO: 4,
                                                    evacc.SCALARS: 0,
                                                    evacc.HISTOGRAMS: 1,
                                                })

#         print('Loading events')
        ea.Reload()  # loads events from file

#         print(ea.Tags())
        df = pd.DataFrame(ea.Scalars('qnet/train_stats/' + action))
        df = df.drop(columns=['wall_time'])
#         print(df.head())

        df.to_csv('{}/{}/{}_{}.csv'.format(logdir, rundir, rundir, action))


def qnet_loss(logdir, rundir):
    more_dir = 'qnet/qnet/train/loss/value'

    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]

    file = os.path.join(logdir, rundir, more_dir, eventsname)

    print('Creating accumulator from {}'.format(os.path.abspath(file)))
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })

#     print('Loading events')
    ea.Reload()  # loads events from file

#     print(ea.Tags())
    df = pd.DataFrame(ea.Scalars('qnet/train/loss'))
    df = df.drop(columns=['wall_time'])
#     print(df.head())

    df.to_csv('{}/{}/{}_loss.csv'.format(logdir, rundir, rundir))


def qnet_dist(logdir, rundir):
    list_stats = ['min', 'avg', 'max', 'sum']
    for which_stat in list_stats:
        more_dir = 'qnet/qnet/val/distance/' + which_stat

        onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                     os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
        eventsname = onlyfiles[0]

        file = os.path.join(logdir, rundir, more_dir, eventsname)

        print('Creating accumulator from {}'.format(os.path.abspath(file)))
        ea = evacc.EventAccumulator(file,
                                                size_guidance={  # see below regarding this argument
                                                    evacc.COMPRESSED_HISTOGRAMS: 500,
                                                    evacc.IMAGES: 4,
                                                    evacc.AUDIO: 4,
                                                    evacc.SCALARS: 0,
                                                    evacc.HISTOGRAMS: 1,
                                                })

#         print('Loading events')
        ea.Reload()  # loads events from file

        if which_stat == 'min':
            df_hits = pd.DataFrame(ea.Scalars('qnet/val/distance'))
            # .to_csv('auc.csv')
#             print(df_hits.head())
#             print(list(df_hits))
            df_hits = df_hits.drop(columns=['wall_time'])
#             print(df_hits.head())
#             print(list(df_hits))
            df_hits = df_hits.rename(index=str, columns={"value": which_stat})
#             print(df_hits.head())
#             print(list(df_hits))
        else:
            df_correct = pd.DataFrame(ea.Scalars('qnet/val/distance'))
#             print(df_correct.head())
#             print(list(df_correct))
            df_hits = df_hits.reset_index(drop=True)
            df_correct = df_correct.reset_index(drop=True)
            df_hits[which_stat] = df_correct['value']
#             print(df_hits.head())

    df_hits.to_csv('{}/{}/{}_distance.csv'.format(logdir, rundir, rundir))


def qnet_hits(logdir, rundir):
    more_dir = 'qnet/qnet/val/hits/total'
    # eventsname = 'events.out.tfevents.1532184516.d66112711c6d'

    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]

    file = os.path.join(logdir, rundir, more_dir, eventsname)

    print('Creating accumulator from {}'.format(os.path.abspath(file)))
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })

#     print('Loading events')
    ea.Reload()  # loads events from file

#     print(ea.Tags())

    df_hits = pd.DataFrame(ea.Scalars('qnet/val/hits'))
    # .to_csv('auc.csv')
#     print(df_hits.head())
#     print(list(df_hits))
    df_hits = df_hits.drop(columns=['wall_time'])
#     print(df_hits.head())
#     print(list(df_hits))
    df_hits = df_hits.rename(index=str, columns={"value": "total"})
#     print(df_hits.head())
#     print(list(df_hits))

    more_dir = 'qnet/qnet/val/hits/correct'

    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]

    file = os.path.join(logdir, rundir, more_dir, eventsname)

#     print('Creating accumulator')
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })

#     print('Loading events')
    ea.Reload()  # loads events from file

#     print(ea.Tags())
    df_correct = pd.DataFrame(ea.Scalars('qnet/val/hits'))
#     print(df_correct.head())
#     print(list(df_correct))
    df_hits = df_hits.reset_index(drop=True)
    df_correct = df_correct.reset_index(drop=True)
    df_hits['correct'] = df_correct['value']
#     print(df_hits.head())

    df_hits.to_csv('{}/{}/{}_hits.csv'.format(logdir, rundir, rundir))

def qnet_confmats(logdir, rundir):
    more_dir = 'qnet/qnet/val/incorrect/fn'
    
    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]
    
    file = os.path.join(logdir, rundir, more_dir, eventsname)
    
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })
    
    ea.Reload()  # loads events from file
    print(ea.Tags())
    df_hits = pd.DataFrame(ea.Scalars('qnet/val/incorrect'))
    print(df_hits.head())
    df_hits = df_hits.drop(columns=['wall_time'])
    df_hits.to_csv('{}/{}/{}_fn.csv'.format(logdir, rundir, rundir))
    
    more_dir = 'qnet/qnet/val/incorrect/fp'
    
    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]
    
    file = os.path.join(logdir, rundir, more_dir, eventsname)
    
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })
    
    ea.Reload()  # loads events from file
    df_hits = pd.DataFrame(ea.Scalars('qnet/val/incorrect'))
    df_hits = df_hits.drop(columns=['wall_time'])
    df_hits.to_csv('{}/{}/{}_fp.csv'.format(logdir, rundir, rundir))
    
    more_dir = 'qnet/qnet/val/correct/tn'
    
    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]
    
    file = os.path.join(logdir, rundir, more_dir, eventsname)
    
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })
    
    ea.Reload()  # loads events from file
    print(ea.Tags())
    df_hits = pd.DataFrame(ea.Scalars('qnet/val/correct'))
    print(df_hits.head())
    df_hits = df_hits.drop(columns=['wall_time'])
    df_hits.to_csv('{}/{}/{}_tn.csv'.format(logdir, rundir, rundir))
    
    
    more_dir = 'qnet/qnet/val/correct/tp'
    
    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]
    
    file = os.path.join(logdir, rundir, more_dir, eventsname)
    
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })
    
    ea.Reload()  # loads events from file
    df_hits = pd.DataFrame(ea.Scalars('qnet/val/correct'))
    df_hits = df_hits.drop(columns=['wall_time'])
    df_hits.to_csv('{}/{}/{}_tp.csv'.format(logdir, rundir, rundir))
    
def qnet_per_img(logdir, rundir):
    more_dir = 'qnet/qnet/val/per_img_hits/value'

    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]

    file = os.path.join(logdir, rundir, more_dir, eventsname)

#     print('Creating accumulator')
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })

#     print('Loading events')
    ea.Reload()  # loads events from file

    df_hits = pd.DataFrame(ea.Scalars('qnet/val/per_img_hits'))
    # .to_csv('auc.csv')
#     print(df_hits.head())
#     print(list(df_hits))
    df_hits = df_hits.drop(columns=['wall_time'])
#     print(df_hits.head())
#     print(list(df_hits))

    df_hits.to_csv('{}/{}/{}_per_img_hits.csv'.format(logdir, rundir, rundir))
    
def qnet_whatever(logdir, rundir):
    more_dir = 'qnet/qnet/val/lesions_acc/value'

    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]

    file = os.path.join(logdir, rundir, more_dir, eventsname)

#     print('Creating accumulator')
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })

#     print('Loading events')
    ea.Reload()  # loads events from file

    df_hits = pd.DataFrame(ea.Scalars('qnet/val/lesions_acc'))
    # .to_csv('auc.csv')
#     print(df_hits.head())
#     print(list(df_hits))
    df_hits = df_hits.drop(columns=['wall_time'])
#     print(df_hits.head())
#     print(list(df_hits))

    df_hits.to_csv('{}/{}/{}_lesions_acc.csv'.format(logdir, rundir, rundir))
    
    
    
    more_dir = 'qnet/qnet/val/score/value'

    onlyfiles = [f for f in os.listdir(os.path.join(logdir, rundir, more_dir)) if
                 os.path.isfile(os.path.join(os.path.join(logdir, rundir, more_dir), f))]
    eventsname = onlyfiles[0]

    file = os.path.join(logdir, rundir, more_dir, eventsname)

#     print('Creating accumulator')
    ea = evacc.EventAccumulator(file,
                                            size_guidance={  # see below regarding this argument
                                                evacc.COMPRESSED_HISTOGRAMS: 500,
                                                evacc.IMAGES: 4,
                                                evacc.AUDIO: 4,
                                                evacc.SCALARS: 0,
                                                evacc.HISTOGRAMS: 1,
                                            })

#     print('Loading events')
    ea.Reload()  # loads events from file

    df_hits = pd.DataFrame(ea.Scalars('qnet/val/score'))
    # .to_csv('auc.csv')
#     print(df_hits.head())
#     print(list(df_hits))
    df_hits = df_hits.drop(columns=['wall_time'])
#     print(df_hits.head())
#     print(list(df_hits))

    df_hits.to_csv('{}/{}/{}_score.csv'.format(logdir, rundir, rundir))

    
def main(logdir, rundir, which_plots=[True,True,True,True,True,True,True,True]):
    # feat(logdir,rundir)
    if which_plots[0]:
        qnet_loss(logdir,rundir)
    if which_plots[1]:
        qnet_q_vals(logdir,rundir)
    if which_plots[2]:
        qnet_eps(logdir,rundir)
    if which_plots[3]:
        qnet_dist(logdir, rundir)
    if which_plots[4]:
        qnet_hits(logdir, rundir)
    if which_plots[5]:
        qnet_per_img(logdir, rundir)
    if which_plots[6]:
        qnet_whatever(logdir, rundir)
    if which_plots[7]:
        qnet_confmats(logdir, rundir)
    pass
 

if __name__ == '__main__':
    logdir = '../../logs_from_18_07_18'
    rundir = 'qnet_simple_05_1'
    do_loss = True
    do_qvals = True
    do_eps = True
    do_dist = True
    do_hits = True
    do_per_img = True
    do_conf_mat = True
    
    main(logdir, rundir,[do_loss, do_qvals, do_eps, do_dist, do_hits, do_per_img, do_conf_mat])
    
