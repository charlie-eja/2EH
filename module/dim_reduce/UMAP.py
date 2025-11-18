import umap
from sklearn.datasets import load_iris
from module.Utilities import  preprocessing
import numpy as np

def  umap_train(data : np.ndarray,
                n_neighbors :int =15,
                min_dist :int =0.1,
                n_components :int =2,
                metric : str ="euclidean",
                random_state :int =42) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state
    )
    data_umap = reducer.fit_transform(data)
    return data_umap

def umap_test():
    x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data =preprocessing.normalize_gaussian(x)
    data_umap=umap_train(normalize_data)



if __name__ == '__main__':
    umap_test()