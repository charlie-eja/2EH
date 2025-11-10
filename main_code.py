import numpy as np
from module.Utilities import normalize ,error_callback
from module.dim_reduce import PCA as PCA_EJ ,UMAP as UMAP_EJ ,tSNE as tSNE_EJ
from module.plot_figure import PCA_plot , UMPA_plot , tSNE_plot
from sklearn.datasets import load_iris

def simulation_pca():
    x, y = load_iris(return_X_y=True)
    normalize_data,mean_data,std_data= normalize.normalize_gaussian(data=x)
    pca_model,pca_vector,pca_variance,pca_variance_ratio,latent_data= PCA_EJ.pca_train(data=normalize_data)
    PCA_plot.pca_plot_2D(data=normalize_data,pca_vector=pca_vector,pca_variance_ratio=pca_variance_ratio)

def simulation_umap():
    x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data = normalize.normalize_gaussian(data=x)
    umap_data=UMAP_EJ.umap_mapping(data=normalize_data,n_components=2,n_neighbors=15,random_state=42,)
    UMPA_plot.umap_plot_2D(data=umap_data,labels=y)

def simulation_tsne():
    x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data = normalize.normalize_gaussian(data=x)
    tSNE_EJ.tsne_mapping(data=normalize_data,n_components=2,perplexity=30,random_state=42,)
    tSNE_plot.tsne_plot_2D(data=normalize_data,labels=y)



def main():
    try:
        simulation_tsne()
    except Exception as e:
        error_callback.print_project_trace(e)
if __name__ == '__main__':
    main()