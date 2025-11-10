import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def tsne_plot_2D(data,labels=None,sample=30,plot_samping=5):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=sample)
    for i in range(0,len(data),plot_samping):
        plt.text(data[i, 0], data[i, 1], str(i),
                 fontsize=8, color='black',)
    plt.title("t-SNE ")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    plt.savefig('t-SNE_plot.png', dpi=150)
    plt.show()

def tsne_plot_3D(data,labels=None,sample=30,plot_samping=5):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # 假設 data 是 shape = (n_samples, 3)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=sample)

    # 加上每隔 plot_samping 的點標註編號
    for i in range(0, len(data), plot_samping):
        ax.text(data[i, 0], data[i, 1], data[i, 2], str(i),
                fontsize=8, color='black')

    ax.set_title("t-SNE 3D")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_zlabel("t-SNE-3")

    plt.tight_layout()
    plt.savefig('t-SNE_plot_3D.png', dpi=150)
    plt.show()
