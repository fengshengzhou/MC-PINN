import numpy as np

# Min-Max Scaling
def min_max_scale(data, feature_range=(0, 1)):

    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    range_min, range_max = feature_range


    constant_columns = np.where(max_vals == min_vals)[0]
    if len(constant_columns) > 0:
        print(f"：{constant_columns + 1}")
        raise ValueError("!")


    scaled_data = (data - min_vals) / (max_vals - min_vals)
    scaled_data = scaled_data * (range_max - range_min) + range_min
    return scaled_data, min_vals, max_vals


def inverse_min_max_scale(scaled_data, min_vals, max_vals, feature_range=(0, 1)):

    range_min, range_max = feature_range
    scaled_data = (scaled_data - range_min) / (range_max - range_min)
    original_data = scaled_data * (max_vals - min_vals) + min_vals
    return original_data

import numpy as np


