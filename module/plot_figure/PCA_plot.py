import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def pca_plot_2D(data,pca_vector,pca_variance_ratio,labels=None,plot_samping=5):
    pca_vector_dimension = pca_vector.shape[1]
    if pca_vector_dimension == 2:
        latent_data = np.matmul(data, pca_vector)
        x = latent_data[:,0]
        y = latent_data[:, 1]
        pca_variance_ratio_x=np.round(pca_variance_ratio[0],2)
        pca_variance_ratio_y=np.round(pca_variance_ratio[1],2)
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y,c=labels, label='Data Points')

        plt.xlabel(f'X  {str(pca_variance_ratio_x)} %')
        plt.ylabel(f'Y  {str(pca_variance_ratio_y)} %')
        plt.title('2D Data Visualization')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        ax = plt.gca()
        for i in range(0, len(x), plot_samping):
            ax.text(x[i], y[i],  str(i), fontsize=8, color='black')

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if (x_max - x_min) < 0.5:
            ax.set_xlim(-0.3, 0.3)
        if (y_max - y_min) < 0.5:
            ax.set_ylim(-0.3, 0.3)

        plt.savefig('pca_plot.png', dpi=150)
        plt.show()
        print('finished plotting')
    else:
        print('pca_vector_dimension must be 2 ')

def pca_plot_3D(data,pca_vector,pca_variance_ratio,labels=None,plot_samping=5):

    latent_data = np.matmul(data, pca_vector)  # (N x 3)
    x = latent_data[:, 0]
    y = latent_data[:, 1]
    z = latent_data[:, 2]

    pca_variance_ratio_x = np.round(pca_variance_ratio[0] , 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)
    pca_variance_ratio_z = np.round(pca_variance_ratio[2], 2)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=labels, label='Data Points')

    ax.set_xlabel(f'PC1  {pca_variance_ratio_x} %')
    ax.set_ylabel(f'PC2  {pca_variance_ratio_y} %')
    ax.set_zlabel(f'PC3  {pca_variance_ratio_z} %')
    ax.set_title('3D PCA Visualization')

    for i in range(0, len(x), plot_samping):
        ax.text(x[i], y[i], z[i], str(i), fontsize=8, color='black')

    ax.grid(True)
    ax.legend()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    if (x_max - x_min) < 0.5:
        ax.set_xlim(-0.3, 0.3)
    if (y_max - y_min) < 0.5:
        ax.set_ylim(-0.3, 0.3)
    if (z_max - z_min) < 0.5:
        ax.set_zlim(-0.3, 0.3)


    plt.tight_layout()
    plt.savefig('pca_plot_3d.png', dpi=150)
    plt.show()
    print('finished 3D plotting')


def plot_test():
    # 假資料
    data = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12],])

if __name__=='__main__':
    plot_test()