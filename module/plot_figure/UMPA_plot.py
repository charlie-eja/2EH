import matplotlib.pyplot as plt


def umap_plot_2D(data,labels=None,sample=30):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=sample)
    plt.title("UMAP ")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig('umap_plot.png', dpi=150)
    plt.show()