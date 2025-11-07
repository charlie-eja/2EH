import numpy as np
from module.Utilities import normalize
from module.dim_reduce import PCA as PCA_EJ
from module.plot_figure import PCA_plot

def main():
    # 假資料
    rng = np.random.default_rng(42)
    n = 150
    data_x1 = rng.normal(0, 1, n)
    data_x2 = 2 * data_x1 + rng.normal(0, 1, n)  # correlated with x1
    data_x3 = 3 * data_x1 +2 *data_x2
    data = np.column_stack([data_x1, data_x2 ,data_x3])
    normalize_data,mean_data,std_data= normalize.normalize_gaussian(data)
    pca_model,pca_vector,pca_variance,pca_variance_ratio,latent_data= PCA_EJ.pca_train(normalize_data, pca_components=2)
    PCA_plot.pca_plot_2D(normalize_data,pca_vector,pca_variance_ratio)

if __name__ == '__main__':
    main()