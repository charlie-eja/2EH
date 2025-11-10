import matplotlib.pyplot as plt

def umap_plot_2D(data,labels=None,sample=30,plot_samping=5):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=sample)
    for i in range(0,len(data),plot_samping):
        plt.text(data[i, 0], data[i, 1], str(i),
                 fontsize=8, color='black',)
    plt.title("UMAP ")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig('umap_plot.png', dpi=150)
    plt.show()

def umap_plot_3D(data,labels=None,sample=30,plot_samping=5):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=sample)
    for i in range(0, len(data), plot_samping):
        ax.text(data[i, 0], data[i, 1], data[i, 2], str(i),
                fontsize=8, color='black')

    ax.set_title("UMAP 3D")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")

    plt.tight_layout()
    plt.savefig('umap_plot_3D.png', dpi=150)
    plt.show()