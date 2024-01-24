import numpy as np

CUT_OFF_THRESHOLD = 10
def get_burst(trace, fdir):
    # first check whether there are some outlier pkts based on the CUT_OFF_THRESHOLD
    # If the outlier index is within 50, cut off the head
    # else, cut off the tail

    
    start, end = 0, len(trace)
    ipt_burst = np.diff(trace[:, 0])
    ipt_outlier_inds = np.where(ipt_burst > CUT_OFF_THRESHOLD)[0]
    """ If an outlier is detected within the first 50 packets, 
    the trace is truncated from the start to this point. If an outlier is found after the first 50 packets, 
    the trace is truncated from this point to the end. """
    if len(ipt_outlier_inds) > 0:
        outlier_ind_first = ipt_outlier_inds[0]
        if outlier_ind_first < 50:
            start = outlier_ind_first + 1
        outlier_ind_last = ipt_outlier_inds[-1]
        if outlier_ind_last > 50:
            end = outlier_ind_last + 1
        

    if start != 0 or end != len(trace):
        print("File {} trace has been truncated from {} to {}".format(fdir, start, end))

    trace = trace[start:end].copy()

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
    return np.array(merged_burst_seqs)
trace = np.array([[ 0.0  ,   1.  ],
 [ 0.2516 , 1.  ],
 [ 0.4695 ,-1.  ],
 [ 0.4718 , 1.  ],
 [ 0.4719 , 1.  ],
 [ 0.7277 ,-1.  ],
 [ 0.7313 , 1.  ],
 [ 0.7314 , 1.  ],
 [ 0.975 , -1.  ],
 [ 0.9751, -1.  ],
 [ 0.9751, -1.  ],
 [ 0.9751, -1.  ]])

burst = get_burst(trace, '')

print(burst)