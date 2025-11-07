import umap
from sklearn.datasets import load_iris
from module.Utilities import  normalize


def  umap_mapping(data,n_neighbors=15,min_dist=0.1,n_components=2,metric="euclidean",random_state=42):
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
    normalize_data, mean_data, std_data =normalize.normalize_gaussian(x)
    data_umap=umap_mapping(normalize_data)



if __name__ == '__main__':
    umap_test()