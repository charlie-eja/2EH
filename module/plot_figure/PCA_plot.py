import matplotlib.pyplot as plt
import numpy as np

def pca_plot_2D(data,pca_vector,pca_variance_ratio,labels=None,plot_samping=-1,figure_name='pca_plot'):

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

    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()
    print('finished plotting')

def pca_plot_3D(data,pca_vector,pca_variance_ratio,labels=None,plot_samping=-1,figure_name='pca_plot_3D'):

    latent_data = np.matmul(data, pca_vector)  # (N x 3)
    x = latent_data[:, 0]
    y = latent_data[:, 1]
    z = latent_data[:, 2]

    pca_variance_ratio_x = np.round(pca_variance_ratio[0] , 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)
    pca_variance_ratio_z = np.round(pca_variance_ratio[2], 2)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=labels, label='Data Points')
    ax.set_xlabel(f'PC1  {pca_variance_ratio_x} %')
    ax.set_ylabel(f'PC2  {pca_variance_ratio_y} %')
    ax.set_zlabel(f'PC3  {pca_variance_ratio_z} %')
    ax.set_title('3D PCA Visualization')
    ax.grid(True)
    ax.legend()

    for i in range(0, len(x), plot_samping):
        ax.text(x[i], y[i], z[i], str(i), fontsize=8, color='black')

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
    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()
    print('finished 3D plotting')

def pca_plot_2D_color(data,pca_vector,pca_variance_ratio,plot_samping=-1,figure_name='pca_plot_colored'):

    latent_data = np.matmul(data, pca_vector)
    x = latent_data[:, 0]
    y = latent_data[:, 1]

    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)

    order_colors = np.arange(len(x))
    plt.figure(figsize=(10, 5))
    sc = plt.scatter(x, y, c=order_colors, cmap='viridis', s=25)
    plt.xlabel(f'X  {pca_variance_ratio_x} %')
    plt.ylabel(f'Y  {pca_variance_ratio_y} %')
    plt.title('2D Data Visualization (Color = Order)')
    plt.colorbar(sc, label='Order / Index')
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()

    for i in range(0, len(x), plot_samping):
        ax.text(x[i], y[i], str(i), fontsize=8, color='black')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if (x_max - x_min) < 0.5:
        ax.set_xlim(-0.3, 0.3)
    if (y_max - y_min) < 0.5:
        ax.set_ylim(-0.3, 0.3)

    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()
    print('finished plotting')

def pca_plot_3D_color(data,pca_vector,pca_variance_ratio,plot_samping=-1,figure_name='pca_plot_3D_colored'):

    latent_data = np.matmul(data, pca_vector)  # (N x 3)
    x = latent_data[:, 0]
    y = latent_data[:, 1]
    z = latent_data[:, 2]

    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)
    pca_variance_ratio_z = np.round(pca_variance_ratio[2], 2)

    order_colors = np.arange(len(x))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=order_colors, cmap='viridis', s=20)
    ax.set_xlabel(f'PC1  {pca_variance_ratio_x} %')
    ax.set_ylabel(f'PC2  {pca_variance_ratio_y} %')
    ax.set_zlabel(f'PC3  {pca_variance_ratio_z} %')
    ax.set_title('3D PCA Visualization (Color = Order)')
    ax.grid(True)

    for i in range(0, len(x), plot_samping):
        ax.text(x[i], y[i], z[i], str(i), fontsize=8, color='black')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()
    if (x_max - x_min) < 0.5:
        ax.set_xlim(-0.3, 0.3)
    if (y_max - y_min) < 0.5:
        ax.set_ylim(-0.3, 0.3)
    if (z_max - z_min) < 0.5:
        ax.set_zlim(-0.3, 0.3)

    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Order / Index')
    plt.tight_layout()
    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()
    print('finished 3D plotting')


def pca_plot_2D_variable_vector(pca_vector,pca_variance_ratio,ax=None, overlab=True):
    if overlab:
        if ax is None:
            raise ValueError("overlab=True need figure information ")
        target_ax = ax
    else:
        plt.figure(figsize=(6, 4))
        target_ax = plt.gca()



    x_line = np.linspace(-1, 1, 100)
    target_ax.plot(x_line, x_line, linestyle='--', label='Extra Plot')

    if not overlab:
        target_ax.legend()
        target_ax.grid(True)
        plt.show()


def plot_test():
    data = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12],])

if __name__=='__main__':
    plot_test()