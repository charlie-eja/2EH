import numpy as np
from module.Utilities import normalize , error_callback , samping_method
from module.dim_reduce import PCA as PCA_EJ ,UMAP as UMAP_EJ ,tSNE as tSNE_EJ
from module.plot_figure import PCA_plot , UMPA_plot , tSNE_plot
from sklearn.datasets import load_iris
import pandas as pd

def simulation_pca(x,y=None):
    # x, y = load_iris(return_X_y=True)
    normalize_data,mean_data,std_data= normalize.normalize_gaussian(data=x)
    pca_model,pca_vector,pca_variance,pca_variance_ratio,latent_data = (
        PCA_EJ.pca_train(data=normalize_data,n_components=3,))
    PCA_plot.pca_plot_3D(
        data=normalize_data,pca_vector=pca_vector,pca_variance_ratio=pca_variance_ratio,plot_samping=-1,labels=y)
    PCA_plot.pca_plot_3D_color(
        data=normalize_data,pca_vector=pca_vector,pca_variance_ratio=pca_variance_ratio,plot_samping=-1)

def simulation_umap(x,y=None):
    # x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data = normalize.normalize_gaussian(data=x)
    umap_data=UMAP_EJ.umap_mapping(data=normalize_data,n_components=3,n_neighbors=15,random_state=42,)
    UMPA_plot.umap_plot_3D(data=umap_data,labels=y,plot_samping=-1,)
    UMPA_plot.umap_plot_3D_color(data=umap_data, plot_samping=-1, )

def simulation_tsne(x,y=None):
    # x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data = normalize.normalize_gaussian(data=x)
    tSNE_EJ.tsne_mapping(data=normalize_data,n_components=3,perplexity=30,random_state=42,)
    tSNE_plot.tsne_plot_3D(data=normalize_data,labels=y,plot_samping=-1,)
    tSNE_plot.tsne_plot_3D_color(data=normalize_data,plot_samping=-1,)

def main():
    # try:
    #     simulation_pca()
    # except Exception as e:
    #     error_callback.print_project_trace(e)
    # try:
    data=pd.read_excel(r'D:\Pycharm Project\2EH\data\Heat_Recovery_System.xlsx',sheet_name='Sheet2')
    # interval_data=samping_method.interval_sampling(data=data,interval_count=5,start_index=1)

    interval_data1 = samping_method.time_sampling(
        data, interval_count=3600, start_time='2023-01-01', end_time='2023-05-11 ')
    interval_data_np1=interval_data1.iloc[:, 1:].to_numpy(dtype=float)

    interval_data2 = samping_method.time_sampling(
        data, interval_count=3600, start_time='2023-05-20', end_time='2023-07-22 ')
    interval_data_np2=interval_data2.iloc[:, 1:].to_numpy(dtype=float)

    interval_data3 = samping_method.time_sampling(
        data, interval_count=3600, start_time='2023-08-13', end_time='2023-12-01 ')
    interval_data_np3=interval_data3.iloc[:, 1:].to_numpy(dtype=float)

    # normalize_data3, mean_data3, std_data3 = normalize.normalize_gaussian(data=interval_data_np3)

    # interval_data_np=interval_data_np2
    interval_data_np=np.vstack((interval_data_np1, interval_data_np2, interval_data_np3))

    simulation_pca(x=interval_data_np)
    simulation_tsne(x=interval_data_np)
    simulation_umap(x=interval_data_np)
    print('end')
    # except Exception as e:
    #     error_callback.print_project_trace(e)


if __name__ == '__main__':
    main()