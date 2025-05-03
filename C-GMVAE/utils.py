import os
import torch
import numpy as np 

def build_path(path):
    path_levels = path.split('/')
    cur_path = ""
    for path_seg in path_levels:
        if len(cur_path):
            cur_path = cur_path + '/' + path_seg
        else: 
            cur_path = path_seg
        
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)

# [meta | label | feature]

def get_label(data, order, offset, label_dim):
    output = []
    for i in order:
        output.append(data[i][offset:offset+label_dim])

    output = np.array(output, dtype='int')
    return output

def get_feat(data, order, meta_offset, label_dim, feature_dim):
    output = []
    meta_output = []
    offset = meta_offset + label_dim
    for i in order:
        meta_output.append(data[i][:meta_offset])
        output.append(data[i][offset:offset+feature_dim])

    output = np.array(output, dtype='float32')
    meta_output = np.array(meta_output, dtype='float32')
    return np.concatenate([output, meta_output], axis=1)
