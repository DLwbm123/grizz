#!/usr/bin/env python3
import functools
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np


def get_cfg_data():
    with open("paramters.yaml", "r") as ymlfile:
        return yaml.load(ymlfile)

def memo(fn):
    '''
        A decorator which will reduce duplicate call.
    '''
    cache = {}
    miss = object()

    @functools.wraps(fn)
    def wrapper(*args):
        k = args or fn.func_code.co_filename + '.' + fn.func_name
        print(k)
        result = cache.get(k, miss)

        if result is miss:
            result = fn(*args)
            cache[k] = result
        return result

    return wrapper

def get_point_index_map(distribution):
    point_index_map = {}
    index = 0
    for point in distribution:
        point_index_map[tuple(point)] = index
        index += 1
    return point_index_map

def get_labels(size_list):
    label_index = 0
    labels = []
    for size_num in size_list:
        labels.extend([label_index] * size_num)
        label_index += 1
    return np.array(labels)
