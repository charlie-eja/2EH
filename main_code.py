import numpy as np
from module.Utilities import normalize , error_callback , samping_method
from module.dim_reduce import PCA as PCA_EJ ,UMAP as UMAP_EJ ,tSNE as tSNE_EJ
from module.plot_figure import PCA_plot , UMPA_plot , tSNE_plot
from sklearn.datasets import load_iris
import pandas as pd


def simulation_pca(data_title,x,y=None):
    # x, y = load_iris(return_X_y=True)
    normalize_data,mean_data,std_data= normalize.normalize_gaussian(data=x)
    pca_model,pca_vector,pca_variance,pca_variance_ratio,latent_data = (
        PCA_EJ.pca_train(data=normalize_data,n_components=2,))
    PCA_plot.pca_plot_2D_html(
        data=normalize_data,pca_vector=pca_vector,pca_variance_ratio=pca_variance_ratio,plot_samping=-1,labels=y)
    PCA_plot.pca_plot_2D_color_html(
        data=normalize_data,pca_vector=pca_vector,pca_variance_ratio=pca_variance_ratio,plot_samping=-1)
    PCA_plot.pca_plot_2D_variable_vector_html(pca_vector=pca_vector,pca_variance_ratio=pca_variance_ratio,
                                              vector_name=data_title)
def simulation_umap(x,y=None):
    # x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data = normalize.normalize_gaussian(data=x)
    umap_data=UMAP_EJ.umap_train(data=normalize_data,n_components=2,n_neighbors=15,random_state=42,)
    UMPA_plot.umap_plot_2D_html(data=umap_data,labels=y,plot_samping=-1,)
    UMPA_plot.umap_plot_2D_color_html(data=umap_data, plot_samping=-1, )

def simulation_tsne(x,y=None):
    # x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data = normalize.normalize_gaussian(data=x)
    tSNE_EJ.tsne_train(data=normalize_data,n_components=2,perplexity=30,random_state=42,)
    tSNE_plot.tsne_plot_2D_html(data=normalize_data,labels=y,plot_samping=-1,)
    tSNE_plot.tsne_plot_2D_color_html(data=normalize_data,plot_samping=-1,)

def main():
    # try:
    #     simulation_pca()
    # except Exception as e:

    #     error_callback.print_project_trace(e)
    # try:
    data=pd.read_excel(r'D:\Pycharm Project\2EH\data\Heat_Recovery_System.xlsx',sheet_name='Sheet2')

    data_title = data.columns.tolist()[1:]
    data_title = [s[6:] for s in data_title]

    interval_data1 = samping_method.time_sampling(
        data, interval_count=3600, start_time='2023-01-01', end_time='2023-05-10 ')
    interval_data_np1=interval_data1.iloc[:, 1:].to_numpy(dtype=float)

    interval_data2 = samping_method.time_sampling(
        data, interval_count=3600, start_time='2023-05-20', end_time='2023-07-22 ')
    interval_data_np2=interval_data2.iloc[:, 1:].to_numpy(dtype=float)

    interval_data3 = samping_method.time_sampling(
        data, interval_count=3600, start_time='2023-08-13', end_time='2023-12-01 ')
    interval_data_np3=interval_data3.iloc[:, 1:].to_numpy(dtype=float)

    # interval_data_np=interval_data_np1
    interval_data_np=np.vstack((interval_data_np1, interval_data_np2, interval_data_np3))

    normalize_data, mean_data, std_data = normalize.normalize_gaussian(data=interval_data_np)

    simulation_pca(data_title,x=normalize_data)
    simulation_tsne(x=normalize_data)
    simulation_umap(x=normalize_data)
    print('end')
    # except Exception as e:
    #     error_callback.print_project_trace(e)


if __name__ == '__main__':
    main()