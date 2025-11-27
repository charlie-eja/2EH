import numpy as np

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

from module.Utilities import preprocessing , error_callback
from module.dim_reduce import PCA as PCA_EJ ,UMAP as UMAP_EJ ,tSNE as tSNE_EJ
from module.plot_figure import PCA_plot , UMPA_plot , TSNE_plot
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
    TSNE_plot.tsne_plot_2D_html(data=normalize_data,labels=y,plot_samping=-1,)
    TSNE_plot.tsne_plot_2D_color_html(data=normalize_data,plot_samping=-1,)

def main():
    # try:
        data = pd.read_excel(r'data\Heat_Recovery_System.xlsx',sheet_name='Sheet2')
        data = preprocessing.find_nan_data(data,max_gap=5)


        input_list = ['ML2EH_TI-141-11A.PV','ML2EH_TI-141-11B.PV','ML2EH_TI-141-11C.PV',
                      'ML2EH_TI-134-1.PV','ML2EH_TI-134-2.PV','ML2EH_TI-126-4.PV',
                      'ML2EH_TI-126-5.PV','ML2EH_TI-126-6.PV','ML2EH_TI-126-7.PV',
                      'ML2EH_TI-127-1.PV','ML2EH_TI-127-2.PV','ML2EH_TI-128-3.PV',
                      'ML2EH_TI-126-8.PV','ML2EH_TI-126-2.PV','ML2EH_FIC-115-3.PV',]
        output_list = ['ML2EH_FI-165-1.PV','ML2EH_TI-128-1.PV','ML2EH_H143-O2',
                       'ML2EH_TIC-126-1.PV','ML2EH_TI-126-3.PV',]

        input_index_list,output_index_list=preprocessing.find_index(data.columns,input_list,output_list)

        # data_title = data.columns.tolist()[1:]
        # data_title = [s[6:] for s in data_title]

        start_time_list=['2023-01-01','2023-05-20','2023-08-13']
        end_time_list  =['2023-05-10','2023-07-22','2023-12-01']

        interval_data ,data_lengths= preprocessing.multi_time_sampling(
            data,
            start_time_list,
            end_time_list,
            interval_count=3600,
            time_index='Time',)

        normalize_data, mean_data, std_data = preprocessing.normalize_gaussian(data=interval_data)

        all_x,all_y=preprocessing.multi_sort_3D_data(normalize_data,
                                                     data_lengths=data_lengths,
                                                     jump_step=2,
                                                     input_index=input_index_list+output_index_list,
                                                     output_index=output_index_list,
                                                     input_time_step=24,
                                                     output_time_step=10)

        # simulation_pca(data_title,x=normalize_data)
        # simulation_tsne(x=normalize_data)
        # simulation_umap(x=normalize_data)

        x_train, x_test, y_train, y_test = train_test_split(
            all_x, all_y, test_size=0.2, random_state=42
        )

        x_hidden = 5
        y_hidden = 5
        seq2seq_model,history=seq2seq.seq2seq_model(x_train,y_train,epochs=100,x_hidden=x_hidden,y_hidden=y_hidden)
        seq2seq_plot.plot_loss(history=history)


        # x_test=all_x
        # y_test=all_y

        y_dim =  y_test.shape[2]
        encoder_model, decoder_model = seq2seq.build_inference_models(seq2seq_model, y_dim, y_hidden)
        x_seq =  x_test  # (N, time_step, x_dim)
        y_now =  y_test[:, 0]  # (N, y_dim)
        y_pred = seq2seq.predict_next(x_seq, y_now, encoder_model, decoder_model, y_dim,time_step=10)

        y_pred = y_pred*std_data[output_index_list]+mean_data[output_index_list]
        y_test =  y_test * std_data[output_index_list] + mean_data[output_index_list]

        seq2seq_plot.plot_prediction( y_test, y_pred, output_list,step_index=1)
        seq2seq_plot.plot_prediction( y_test, y_pred, output_list, step_index=9)

        print('end')
    # except Exception as e:
    #     error_callback.print_project_trace(e)


if __name__ == '__main__':
    main()