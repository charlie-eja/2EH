import numpy as np
import normalize
import algorithm_dimension
import plot_figure

def main():
    # 假資料
    rng = np.random.default_rng(42)
    n = 150
    data_x1 = rng.normal(0, 1, n)
    data_x2 = 2 * data_x1 + rng.normal(0, 0.3, n)  # correlated with x1
    data_x3 = 3 * data_x1 +2 *data_x2
    data = np.column_stack([data_x1, data_x2 ,data_x3])
    normalize_data,mean_data,std_data=normalize.normalize_gaussian(data)
    pca_model,pca_vector,pca_variance,pca_variance_ratio,latent_data= algorithm_dimension.pca_train(normalize_data,pca_components=3)
    plot_figure.pca_plot(normalize_data,pca_vector,pca_variance_ratio)
    # int('s')


if __name__ == '__main__':
    main()