import numpy as np

from sklearn.datasets import load_iris
import pandas as pd

from module.Utilities import preprocessing , error_callback
from module.dim_reduce import PCA as PCA_EJ ,UMAP as UMAP_EJ ,tSNE as tSNE_EJ
from module.plot_figure import PCA_plot , UMPA_plot , tSNE_plot
from module.nn_regression import seq2seq
from module.nn_plot_figure import seq2seq_plot


def simulation_pca(data_title,x,y=None):
    # x, y = load_iris(return_X_y=True)
    normalize_data,mean_data,std_data= preprocessing.normalize_gaussian(data=x)
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
    normalize_data, mean_data, std_data = preprocessing.normalize_gaussian(data=x)
    umap_data=UMAP_EJ.umap_train(data=normalize_data,n_components=2,n_neighbors=15,random_state=42,)
    UMPA_plot.umap_plot_2D_html(data=umap_data,labels=y,plot_samping=-1,)
    UMPA_plot.umap_plot_2D_color_html(data=umap_data, plot_samping=-1, )
def simulation_tsne(x,y=None):
    # x, y = load_iris(return_X_y=True)
    normalize_data, mean_data, std_data = preprocessing.normalize_gaussian(data=x)
    tSNE_EJ.tsne_train(data=normalize_data,n_components=2,perplexity=30,random_state=42,)
    tSNE_plot.tsne_plot_2D_html(data=normalize_data,labels=y,plot_samping=-1,)
    tSNE_plot.tsne_plot_2D_color_html(data=normalize_data,plot_samping=-1,)

def main():
    # try:
        data = pd.read_excel(r'D:\Pycharm Project\2EH\data\Heat_Recovery_System.xlsx',sheet_name='Sheet2')
        data = preprocessing.find_nan_data(data,max_gap=5)
        data = data.apply(pd.to_numeric, errors='coerce')

        data_title = data.columns.tolist()[1:]
        data_title = [s[6:] for s in data_title]

        start_time_list=['2023-01-01','2023-05-20','2023-08-13']
        end_time_list  =['2023-05-10','2023-07-22','2023-12-01']

        interval_data ,data_lengths= preprocessing.multi_time_sampling(
            data,
            start_time_list,
            end_time_list,
            interval_count=1800,
            time_index='Time',
            time_low=True)

        normalize_data, mean_data, std_data = preprocessing.normalize_gaussian(data=interval_data)
        all_x,all_y=preprocessing.multi_sort_3D_data(normalize_data,
                                                     data_lengths=data_lengths,
                                                     jump_step=2,)

        simulation_pca(data_title,x=normalize_data)
        # simulation_tsne(x=normalize_data)
        # simulation_umap(x=normalize_data)

        seq2seq_model,history=seq2seq.seq2seq_model(all_x,all_y)
        seq2seq_plot.plot_loss(history=history)

        print('end')
    # except Exception as e:
    #     error_callback.print_project_trace(e)


if __name__ == '__main__':
    main()