import numpy as np
from module.Utilities import normalize
from module.dim_reduce import PCA as PCA_EJ ,UMAP as UMAP_EJ
from module.plot_figure import PCA_plot , UMPA_plot
from sklearn.datasets import load_iris

def simulation_pca():
    rng = np.random.default_rng(42)
    n = 150
    data_x1 = rng.normal(0, 1, n)
    data_x2 = 2 * data_x1 + rng.normal(0, 1, n)  # correlated with x1
    data_x3 = 3 * data_x1 +2 *data_x2
    data = np.column_stack([data_x1, data_x2 ,data_x3])
    normalize_data,mean_data,std_data= normalize.normalize_gaussian(data=data)
    pca_model,pca_vector,pca_variance,pca_variance_ratio,latent_data= PCA_EJ.pca_train(data=normalize_data)
    PCA_plot.pca_plot_2D(data=normalize_data,pca_vector=pca_vector,pca_variance_ratio=pca_variance_ratio)
def simulation_umap():
    x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data = normalize.normalize_gaussian(data=x)
    umap_data=UMAP_EJ.umap_mapping(data=normalize_data,random_state=42,n_neighbors=15)
    UMPA_plot.umap_plot_2D(data=umap_data,labels=y)

def main():
    simulation_pca()
if __name__ == '__main__':
    main()