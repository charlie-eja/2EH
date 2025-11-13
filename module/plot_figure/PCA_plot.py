import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def pca_plot_2D(data,pca_vector,pca_variance_ratio,labels=None,plot_samping=-1,figure_name='pca_plot'):

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

def pca_plot_2D_html(data,pca_vector,pca_variance_ratio,labels=None,plot_samping=-1,figure_name='pca_plot'):
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

def pca_plot_3D_html(data,pca_vector,pca_variance_ratio,labels=None,plot_samping=-1,figure_name='pca_plot_3D'):

    latent_data = np.matmul(data, pca_vector)  # (N x 3)
    x = latent_data[:, 0]
    y = latent_data[:, 1]
    z = latent_data[:, 2]

    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)
    pca_variance_ratio_z = np.round(pca_variance_ratio[2], 2)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=labels,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Label')
            ),
            name='Data Points'
        )
    )

    indices = np.arange(0, len(x), plot_samping)
    fig.add_trace(
        go.Scatter3d(
            x=x[indices],
            y=y[indices],
            z=z[indices],
            mode='text',
            text=[str(i) for i in indices],
            textposition='top center',
            showlegend=False
        )
    )

    fig.update_layout(
        title='3D PCA Visualization',
        scene=dict(
            xaxis_title=f'PC1  {pca_variance_ratio_x} %',
            yaxis_title=f'PC2  {pca_variance_ratio_y} %',
            zaxis_title=f'PC3  {pca_variance_ratio_z} %',
        ),
        margin=dict(l=40, r=20, t=60, b=40)
        # margin=dict(l=0, r=0, b=0, t=40)
    )

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    scene = {}
    if (x_max - x_min) < 0.5:
        scene.setdefault('xaxis', {})['range'] = [-0.3, 0.3]
    if (y_max - y_min) < 0.5:
        scene.setdefault('yaxis', {})['range'] = [-0.3, 0.3]
    if (z_max - z_min) < 0.5:
        scene.setdefault('zaxis', {})['range'] = [-0.3, 0.3]

    if scene:
        fig.update_layout(scene=scene)

    html_name = 'html/' + figure_name + '.html'
    fig.write_html(html_name)
    print('finished 3D plotting, saved as', html_name)


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


def pca_plot_2D_color_html(data,pca_vector,pca_variance_ratio,plot_samping=-1,figure_name='pca_plot_colored'):
    latent_data = np.matmul(data, pca_vector)
    x = latent_data[:, 0]
    y = latent_data[:, 1]

    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)

    order_colors = np.arange(len(x))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=6,
            color=order_colors,
            colorscale='Viridis',
            colorbar=dict(title='Order / Index')
        ),
        showlegend=False
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
            title=f'X  {pca_variance_ratio_x} %',
            range=x_range,
            showgrid=True
        ),
        yaxis=dict(
            title=f'Y  {pca_variance_ratio_y} %',
            range=y_range,
            showgrid=True
        ),
        title='2D Data Visualization (Color = Order)',
        margin=dict(l=40, r=20, t=60, b=40)
    )

    html_name = 'html/'+figure_name + '.html'
    fig.write_html(html_name)
    print('finished plotting, saved', html_name)


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

def pca_plot_3D_color_html(data,pca_vector,pca_variance_ratio,plot_samping=-1,figure_name='pca_plot_3D_colored'):

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

    indices = np.arange(0, len(x), plot_samping)
    fig.add_trace(go.Scatter3d(
        x=x[indices],
        y=y[indices],
        z=z[indices],
        mode='text',
        text=[str(i) for i in indices],
        textposition='top center',
        showlegend=False
    ))

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    scene = {}

    if (x_max - x_min) < 0.5:
        scene.setdefault('xaxis', {})['range'] = [-0.3, 0.3]
    if (y_max - y_min) < 0.5:
        scene.setdefault('yaxis', {})['range'] = [-0.3, 0.3]
    if (z_max - z_min) < 0.5:
        scene.setdefault('zaxis', {})['range'] = [-0.3, 0.3]

    fig.update_layout(
        title='3D PCA Visualization (Color = Order)',
        scene=dict(
            xaxis_title=f'PC1 {pca_variance_ratio_x} %',
            yaxis_title=f'PC2 {pca_variance_ratio_y} %',
            zaxis_title=f'PC3 {pca_variance_ratio_z} %',
            **scene
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    html_name = 'html/' + figure_name + '.html'
    fig.write_html(html_name)
    print('finished 3D plotting, saved as', html_name)

def pca_plot_2D_variable_vector(pca_vector,pca_variance_ratio,vector_name):
    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)

    pca_vector_max=np.max(pca_vector,axis=0)
    pca_vector_plot_points=pca_vector/pca_vector_max/1.05
    origin_x, origin_y = 0, 0
    plt.figure(figsize=(6, 4))
    for i, (x, y) in enumerate(pca_vector_plot_points, start=0):
        plt.plot([origin_x, x], [origin_y, y],'k')
        plt.text(x, y, str(vector_name[i]), fontsize=8, color='red')
    xs, ys = zip(*pca_vector_plot_points)
    plt.xlabel(f'PC1  {str(pca_variance_ratio_x)} %')
    plt.ylabel(f'PC2  {str(pca_variance_ratio_y)} %')
    plt.scatter(xs, ys, color='blue')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.show()

def pca_plot_2D_variable_vector_html(pca_vector,pca_variance_ratio,vector_name,figure_name='pca_plot_vector'):
    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)

    pca_vector_max = np.max(pca_vector, axis=0)
    pca_vector_plot_points = pca_vector / pca_vector_max / 1.05

    origin_x, origin_y = 0, 0

    fig = go.Figure()

    for i, (x, y) in enumerate(pca_vector_plot_points, start=0):
        fig.add_trace(go.Scatter(
            x=[origin_x, x],
            y=[origin_y, y],
            mode='lines',
            line=dict(color='black'),
            showlegend=False
        ))

    xs, ys = zip(*pca_vector_plot_points)

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='markers+text',
        text=[str(v) for v in vector_name],
        textposition='top center',
        marker=dict(color='blue'),
        showlegend=False
    ))

    fig.update_layout(
        xaxis_title=f'PC1  {pca_variance_ratio_x} %',
        yaxis_title=f'PC2  {pca_variance_ratio_y} %',
        xaxis=dict(range=[-1.1, 1.1]),
        yaxis=dict(range=[-1.1, 1.1]),
    )

    html_name = 'html/'+figure_name+'.html'
    fig.write_html(html_name)
    print('saved', html_name)


def pca_plot_3D_variable_vector_html(pca_vector,pca_variance_ratio,vector_name,figure_name='pca_plot_3D_vector'):
    pca_variance_ratio_x = np.round(pca_variance_ratio[0], 2)
    pca_variance_ratio_y = np.round(pca_variance_ratio[1], 2)
    pca_variance_ratio_z = np.round(pca_variance_ratio[2], 2)

    pca_vector_max = np.max(pca_vector, axis=0)
    pca_vector_plot_points = pca_vector / pca_vector_max / 1.05

    origin_x, origin_y, origin_z = 0, 0, 0

    fig = go.Figure()

    for i, (x, y, z) in enumerate(pca_vector_plot_points, start=0):
        fig.add_trace(go.Scatter3d(
            x=[origin_x, x],
            y=[origin_y, y],
            z=[origin_z, z],
            mode='lines',
            line=dict(color='black'),
            showlegend=False
        ))

    xs, ys, zs = zip(*pca_vector_plot_points)

    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers+text',
        text=[str(v) for v in vector_name],
        textposition='top center',
        marker=dict(color='blue'),
        showlegend=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=f'PC1  {pca_variance_ratio_x} %', range=[-1.1, 1.1]),
            yaxis=dict(title=f'PC2  {pca_variance_ratio_y} %', range=[-1.1, 1.1]),
            zaxis=dict(title=f'PC3  {pca_variance_ratio_z} %', range=[-1.1, 1.1])
        ),
        title='3D PCA Vector Plot',
        margin=dict(l=40, r=20, t=60, b=40)
        # margin=dict(l=0, r=0, t=40, b=0)
    )

    html_name = 'html/'+figure_name+'.html'
    fig.write_html(html_name)
    print('saved', html_name)




def plot_test():
    data = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12],])

if __name__=='__main__':
    plot_test()