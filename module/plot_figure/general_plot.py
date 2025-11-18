import matplotlib.pyplot as plt
import numpy as np

def general_plot_2D(data : np.ndarray,labels : np.ndarray=None,plot_samping : int =-1):
    plt.figure(figsize=(6, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels)

    for i in range(0,len(data),plot_samping):
        plt.text(data[i, 0], data[i, 1], str(i),
                 fontsize=8, color='black',)

    plt.title("general figure")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig('general.png', dpi=150)
    plt.show()

def general_plot_2D_color(data : np.ndarray,plot_samping : int =-1):
    plt.figure(figsize=(6, 5))
    order_colors = np.arange(len(data))
    sc = plt.scatter(data[:, 0], data[:, 1], c=order_colors, cmap='viridis')

    for i in range(0, len(data), plot_samping):
        plt.text(data[i, 0], data[i, 1], str(i),
                 fontsize=8, color='black')

    plt.title("general plot (Color = Order)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    cbar = plt.colorbar(sc)
    cbar.set_label('Order / Index')
    plt.savefig('general_plot_colored.png', dpi=150)
    plt.show()

def general_plot_3D(data : np.ndarray,labels : np.ndarray=None,plot_samping : int =-1):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels,)

    for i in range(0, len(data), plot_samping):
        ax.text(data[i, 0], data[i, 1], data[i, 2], str(i),
                fontsize=8, color='black')

    ax.set_title("general plot 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig('general_plot_3D.png', dpi=150)
    plt.show()

def general_plot_3D_color(data : np.ndarray,plot_samping : int =-1):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    order_colors = np.arange(len(data))
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                    c=order_colors, cmap='viridis', s=20)

    for i in range(0, len(data), plot_samping):
        ax.text(data[i, 0], data[i, 1], data[i, 2],
                str(i), fontsize=8, color='black')

    ax.set_title("general plot 3D (Color = Order)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Order / Index')
    plt.tight_layout()
    plt.savefig('general_plot_3d_colored.png', dpi=150)
    plt.show()

