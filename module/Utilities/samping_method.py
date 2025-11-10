import numpy as np

def interval_sampling(data,start_index,interval_count)-> np.ndarray:
    interval_data=data[start_index::interval_count]
    return interval_data

