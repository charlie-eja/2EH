import matplotlib.pyplot as plt
import numpy as np

def umap_plot_2D(data,labels=None,plot_samping=-1,figure_name='umap_plot'):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels)

    for i in range(0,len(data),plot_samping):
        plt.text(data[i, 0], data[i, 1], str(i),
                 fontsize=8, color='black',)

    plt.title("UMAP ")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()

def umap_plot_2D_color(data,plot_samping=-1,figure_name='umap_plot_colored'):
    plt.figure(figsize=(6, 5))
    order_colors = np.arange(len(data))
    sc = plt.scatter(data[:, 0], data[:, 1], c=order_colors, cmap='viridis')

    for i in range(0, len(data), plot_samping):
        plt.text(data[i, 0], data[i, 1], str(i),
                 fontsize=8, color='black')

    plt.title("UMAP Visualization (Color = Order)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.tight_layout()
    cbar = plt.colorbar(sc)
    cbar.set_label('Order / Index')
    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()
    print('finished UMAP plotting')

def umap_plot_3D(data,labels=None,plot_samping=-1,figure_name='umap_plot_3D'):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels,)

    for i in range(0, len(data), plot_samping):
        ax.text(data[i, 0], data[i, 1], data[i, 2], str(i),
                fontsize=8, color='black')

    ax.set_title("UMAP 3D")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    plt.tight_layout()
    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()

def umap_plot_3D_color(data,plot_samping=-1,figure_name='umap_plot_3D_colored'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    order_colors = np.arange(len(data))
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                    c=order_colors, cmap='viridis', s=20)

    for i in range(0, len(data), plot_samping):
        ax.text(data[i, 0], data[i, 1], data[i, 2],
                str(i), fontsize=8, color='black')

    ax.set_title("UMAP 3D (Color = Order)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.grid(True)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Order / Index')
    plt.tight_layout()
    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()
    print('finished UMAP 3D plotting')