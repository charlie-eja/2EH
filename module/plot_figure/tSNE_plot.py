import matplotlib.pyplot as plt

def tsne_plot_2D(data,labels=None,sample=30):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=sample)
    plt.title("T-SNE ")
    plt.xlabel("T-SNE-1")
    plt.ylabel("T-SNE-2")
    plt.tight_layout()
    plt.savefig('T-SNE_plot.png', dpi=150)
    plt.show()