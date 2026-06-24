import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
def pca_plot_2D_color_html(data : np.ndarray,
                            pca_vector : np.ndarray,
                            pca_variance_ratio : np.ndarray,
                            plot_samping : int =-1,
                            figure_name : str ='pca_plot_colored'):
    latent_data = np.matmul(data, pca_vector)  # (N x 2)
    html_name = 'html/'+figure_name + '.html'
    fig.write_html(html_name)
    print('finished plotting, saved', html_name)

def pca_plot_3D_color(data : np.ndarray,
                      pca_vector : np.ndarray,
                      pca_variance_ratio : np.ndarray,
                      plot_samping : int =-1,
                      figure_name : str ='pca_plot_3D_colored'):
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
    plt.tight_layout()
    plt.savefig(figure_name+'.png', dpi=150)
    plt.show()
    print('finished 3D plotting')

def pca_plot_3D_color_html(data : np.ndarray,
                           pca_vector : np.ndarray,
                           pca_variance_ratio : np.ndarray,
                           plot_samping : int =-1,
                           figure_name : str ='pca_plot_3D_colored'):
    latent_data = np.matmul(data, pca_vector)
    x = latent_data[:, 0]
    y = latent_data[:, 1]
    z = latent_data[:, 2]
    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)
    pca_variance_ratio_z = np.round(pca_variance_ratio[2], 2)
    order_colors = np.arange(len(x))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
    x=x,
    y=y,
    z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=order_colors,
            colorscale='Viridis',
            opacity=0.85,
            colorbar=dict(title='Order / Index')
        )
    ))
    fig.update_layout(
        title='3D PCA Visualization (Color = Order)',
        scene=dict(
            xaxis_title=f'PC1 {pca_variance_ratio_x} %',
            yaxis_title=f'PC2 {pca_variance_ratio_y} %',
            zaxis_title=f'PC3 {pca_variance_ratio_z} %',
    html_name = 'html/' + figure_name + '.html'
    fig.write_html(html_name)
    print('finished 3D plotting, saved as', html_name)
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def pca_plot_2D(data : np.ndarray,
                pca_vector : np.ndarray,
                pca_variance_ratio : np.ndarray,
                labels : np.ndarray=None,
                plot_samping : int =-1,
                figure_name : str ='pca_plot'):

    latent_data = np.matmul(data, pca_vector)
    x = latent_data[:,0]
    y = latent_data[:, 1]

    pca_variance_ratio_x=np.round(pca_variance_ratio[0],2)
    pca_variance_ratio_y=np.round(pca_variance_ratio[1],2)

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y,c=labels, label='Data Points')
    plt.xlabel(f'PC1  {str(pca_variance_ratio_x)} %')
    plt.ylabel(f'PC2  {str(pca_variance_ratio_y)} %')
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

def pca_plot_2D_html(data : np.ndarray,
                     pca_vector : np.ndarray,
                     pca_variance_ratio : np.ndarray,
                     labels : np.ndarray=None,
                     plot_samping : int =-1,
                     figure_name : str ='pca_plot'):
    latent_data = np.matmul(data, pca_vector)
    x = latent_data[:, 0]
    y = latent_data[:, 1]

    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=6,
            color=labels,
            colorscale='Viridis',
            colorbar=dict(title='Label')
        ),
        name='Data Points'
    ))

    indices = np.arange(0, len(x), plot_samping)

    fig.add_trace(go.Scatter(
        x=x[indices],
        y=y[indices],
        mode='text',
        text=[str(i) for i in indices],
        textposition='top center',
        showlegend=False
    ))

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = None
    y_range = None

    if (x_max - x_min) < 0.5:
        x_range = [-0.3, 0.3]
    if (y_max - y_min) < 0.5:
        y_range = [-0.3, 0.3]

    fig.update_layout(
        xaxis=dict(
            title=f'PC1  {pca_variance_ratio_x} %',
            range=x_range,
            showgrid=True
        ),
        yaxis=dict(
            title=f'PC2  {pca_variance_ratio_y} %',
            range=y_range,
            showgrid=True
        ),
        title='2D Data Visualization',
        margin=dict(l=40, r=20, t=60, b=40)
    )

    html_name = 'html/'+figure_name + '.html'
    fig.write_html(html_name)
    print('finished plotting, saved', html_name)
