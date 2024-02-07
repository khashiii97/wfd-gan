import numpy as np
import os
from os.path import join
import common as cm
import utils
import argparse
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
logger = utils.init_logger('extract')

# it is possible the trace has a long tail
# if there is a time gap between two bursts larger than CUT_OFF_THRESHOULD
# We cut off the trace here sicne it could be a long timeout or
# maybe the loading is already finished
# Set a very conservative value
CUT_OFF_THRESHOLD = 10
def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract burst sequences ipt from raw traces')

    parser.add_argument('--dir',
                        metavar='<traces path>',
                        required=True,
                        help='Path to the directory with the traffic traces to be simulated.')
    parser.add_argument('--length',
                        type=int,
                        default=1400,
                        help='Pad to length.'
                        )
    parser.add_argument('--norm',
                        type=bool,
                        default=False,
                        help='Shall we normalize burst sizes by dividing it with cell size?'
                        )
    parser.add_argument('--norm_cell',
                        type=bool,
                        default=False,
                        help='Shall we ignore the value of a cell (e.g., +-888 -> +-1)?'
                        )
    parser.add_argument('--format',
                        metavar='<file suffix>',
                        default=".pkt",
                        )
    parser.add_argument('--log',
                        type=str,
                        dest="log",
                        metavar='<log path>',
                        default='stdout',
                        help='path to the log file. It will print to stdout by default.')
    parser.add_argument('--dsname',
                        type=str,
                        default='ds19',
                        help='The name of the used dataset')

    # Parse arguments
    args = parser.parse_args()
    return args


def get_burst(trace, fdir):
    # first check whether there are some outlier pkts based on the CUT_OFF_THRESHOLD
    # If the outlier index is within 50, cut off the head
    # else, cut off the tail
    start, end = 0, len(trace)
    ipt_burst = np.diff(trace[:, 0])
    ipt_outlier_inds = np.where(ipt_burst > CUT_OFF_THRESHOLD)[0]
    original_length = len(trace)
    outliers = []
    original_trace_size = len(trace)
    
    if len(ipt_outlier_inds) > 0:
        outlier_ind_first = ipt_outlier_inds[0]
        if outlier_ind_first < 50:
            start = outlier_ind_first + 1
            outliers.append(outlier_ind_first)
        outlier_ind_last = ipt_outlier_inds[-1]
        if outlier_ind_last > 50:
            end = outlier_ind_last + 1
            outliers.append(outlier_ind_last)

    if start != 0 or end != len(trace):
        print("File {} with length {} had outliers {} in trace has been truncated from {} to {}".format(fdir,original_length, outliers, start, end))

    trace = trace[start:end].copy()
    modified_trace_size = len(trace)
    # remove the first few lines that are incoming packets
    start = -1
    for time, size in trace:
        start += 1
        if size > 0:
            break

    trace = trace[start:].copy()
    trace[:, 0] -= trace[0, 0]
    assert trace[0, 0] == 0
    burst_seqs = trace

    # merge bursts from the same direction
    merged_burst_seqs = []
    cnt = 0
    sign = np.sign(burst_seqs[0, 1])
    time = burst_seqs[0, 0]
    for cur_time, cur_size in burst_seqs:
        if np.sign(cur_size) == sign:
            cnt += cur_size
        else:
            merged_burst_seqs.append([time, cnt])
            sign = np.sign(cur_size)
            cnt = cur_size
            time = cur_time
    merged_burst_seqs.append([time, cnt])
    merged_burst_seqs = np.array(merged_burst_seqs)
    assert sum(merged_burst_seqs[::2, 1]) == sum(trace[trace[:, 1] > 0][:, 1])
    assert sum(merged_burst_seqs[1::2, 1]) == sum(trace[trace[:, 1] < 0][:, 1])
    return np.array(merged_burst_seqs), original_trace_size, modified_trace_size


def extract(trace, fdir):
    global length, norm
    burst_seq, original_trace_size, modified_trace_size = get_burst(trace, fdir)
    times = burst_seq[:, 0]
    bursts = abs(burst_seq[:, 1])
    if norm:
        bursts /= cm.CELL_SIZE
    bursts = list(bursts)
    bursts.insert(0, len(bursts))
    original_burst_size = len(bursts)
    bursts = bursts[:length] + [0] * (length - len(bursts))
    assert len(bursts) == length
    return bursts, times, original_burst_size, original_trace_size, modified_trace_size


def parallel(flist, n_jobs=70):
    with mp.Pool(n_jobs) as p:
        res = p.map(extractfeature, flist)
        p.close()
        p.join()
    return res

# parallel gets the list of cell paths, creates processes and calls extractfeature on each cell path
# extractfeature loads the trace as an np array with form of [[t1,d1], [t2,d2],...] and calls extract on that trace
# extract calls get_burst on the trace to get the trace in burst mode. then it outputs two elements : bursts and time
# bursts: list containing the size of consecutive bursts. the first element is the number of actual bursts. then, 
# the abs of the sizes are added. then, 0 is added until the length of burst is length. so if length is 8, burst will be something like:
# [6, 2.0, 1.0, 2.0, 1.0, 2.0, 4.0, 0, 0].
# time is the start time of each real burst. e.g., [0.     0.4695 0.4718 0.7277 0.7313 0.975 ]
# finally, extractfeature adds a label of the cell, and outputs bursts, times, label
def extractfeature(fdir):
    global MON_SITE_NUM, norm_cell
    fname = fdir.split('/')[-1].split(".")[0]
    trace = utils.loadTrace(fdir, norm_cell=norm_cell)
    bursts, times, original_burst_size, original_trace_size, modified_trace_size = extract(trace, fdir)
    if '-' in fname:
        label = int(fname.split('-')[0])
    else:
        label = int(MON_SITE_NUM)
    return bursts, times, label, original_burst_size, original_trace_size, modified_trace_size


if __name__ == '__main__':
    global MON_SITE_NUM, length, norm, norm_cell
    # parser config and arguments
    args = parse_arguments()
    length = args.length + 1  # add another feature as the real length of the trace
    norm = args.norm
    dsname = args.dsname
    norm_cell = args.norm_cell
    logger.info("Arguments: %s" % (args))
    outputdir = join(cm.outputdir, os.path.split(args.dir.rstrip('/'))[1], 'feature') #rstrip removes the /s at the end of a path
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    cf = utils.read_conf(cm.confdir)
    MON_SITE_NUM = int(cf['monitored_site_num'])
    MON_INST_NUM = int(cf['monitored_inst_num'])
    MON_SITE_START_IND = int(cf['monitored_site_start_ind'])
    MON_INST_START_IND = int(cf['monitored_inst_start_ind'])
    
                                                   
    # if cf['open_world'] == '1':
    #     UNMON_SITE_NUM = int(cf['unmonitored_site_num'])
    #     OPEN_WORLD = 1
    # else:
    #     OPEN_WORLD = 0
    #     UNMON_SITE_NUM = 0

    # logger.info('Extracting features...')

    flist = []
    for i in range(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM):
        for j in range(MON_INST_START_IND, MON_INST_START_IND + MON_INST_NUM):
            if os.path.exists(os.path.join(args.dir, str(i) + "-" + str(j) + args.format)):
                flist.append(os.path.join(args.dir, str(i) + "-" + str(j) + args.format))
    # do not support open world set.
    # for i in range(UNMON_SITE_NUM):
    #     if os.path.exists(os.path.join(args.dir, str(i) + args.format)):
    #         flist.append(os.path.join(args.dir, str(i) + args.format))
    logger.info('In total {} files.'.format(len(flist)))
    raw_data_dict = parallel(flist)
    bursts, times, labels, original_burst_sizes, original_trace_sizes, modified_trace_sizes = zip(*raw_data_dict) 
    """ In summary, this line restructures the data so that all 
    bursts are grouped together in one tuple, all times are grouped together in another tuple, 
    and all labels are in a third tuple, ...  """

    outputdir_stats = join(cm.outputdir, os.path.split(args.dir.rstrip('/'))[1], 'stats') #rstrip removes the /s at the end of a path
    if not os.path.exists(outputdir_stats):
        os.makedirs(outputdir_stats)
    
    length = str(length)
    plt.hist(original_trace_sizes, bins='auto')  # 'auto' lets matplotlib decide the number of bins
    plt.title('Histogram of Trace lengths of ' + dsname + ' With burst size of ' + length)
    plt.xlabel('Trace Lenghts')
    plt.ylabel('Frequency')
    plt.savefig(join(outputdir_stats , 'original_traces.png'))
    #plt.show()   

    plt.hist(modified_trace_sizes, bins='auto')  # 'auto' lets matplotlib decide the number of bins
    plt.title('Histogram of Modified Trace lengths of '  + dsname + ' With burst size of ' + length)
    plt.xlabel('Trace Lenghts')
    plt.ylabel('Frequency')
    plt.savefig(join(outputdir_stats , 'modified_traces.png'))
    #plt.show()   
      

    plt.hist(original_burst_sizes, bins='auto')  # 'auto' lets matplotlib decide the number of bins
    plt.title('Histogram of burst lengths' +dsname + ' With burst size of ' + length)
    plt.xlabel('Burst Lenghts')
    plt.ylabel('Frequency')
    plt.savefig(join(outputdir_stats ,'original_bursts.png'))
    #plt.show()       
    bursts = np.array(bursts)
    labels = np.array(labels)
    logger.info("feature sizes:{}, label size:{}".format(bursts.shape, labels.shape))
    np.savez_compressed(
        join(outputdir, "raw_feature_{}-{}x{}-{}.npz".format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM,
                                                             MON_INST_START_IND, MON_INST_NUM + MON_INST_START_IND)),
        features=bursts, labels=labels)
    logger.info("output to {}".format(join(outputdir, "raw_feature_{}-{}x{}-{}.npz".
                                           format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM,
                                                  MON_INST_START_IND, MON_INST_NUM + MON_INST_START_IND))))

    # save the time information. The even indexes are outgoing timestamps and the odd indexes are incoming ones.
    np.savez(join(outputdir, "time_feature_{}-{}x{}-{}.npz").
             format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM, MON_INST_START_IND,
                    MON_INST_NUM + MON_INST_START_IND), *times)
    


npz_file = np.load(join(outputdir, "raw_feature_{}-{}x{}-{}.npz".format(MON_SITE_START_IND, MON_SITE_START_IND + MON_SITE_NUM,
                                                             MON_INST_START_IND, MON_INST_NUM + MON_INST_START_IND)))

# Access the saved arrays using their respective keys
features = npz_file['features']
labels = npz_file['labels']

print(features.shape)

zero_counts = [np.count_nonzero(row == 0) for row in features]
plt.hist(zero_counts, bins='auto')  # 'auto' lets matplotlib decide the number of bins
plt.title('Histogram of Zero Padding in burst sequences ' + dsname + ' With burst size of ' + length)
plt.xlabel('Number of Zeros Added')
plt.ylabel('Frequency')
plt.savefig(join(outputdir_stats, 'zero_traces.png'))
#plt.show()
