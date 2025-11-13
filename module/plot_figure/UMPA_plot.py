import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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

def umap_plot_2D_html(data,labels=None,plot_samping=-1,figure_name='umap_plot'):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data[:, 0],
        y=data[:, 1],
        mode='markers',
        marker=dict(
            size=6,
            color=labels,
            colorscale='Viridis',
            colorbar=dict(title='Label')
        ),
        showlegend=False
    ))

    indices = np.arange(0, len(data), plot_samping)

    fig.add_trace(go.Scatter(
        x=data[indices, 0],
        y=data[indices, 1],
        mode='text',
        text=[str(i) for i in indices],
        textposition='top center',
        showlegend=False
    ))

    fig.update_layout(
        title='UMAP',
        xaxis=dict(title='UMAP-1', showgrid=True),
        yaxis=dict(title='UMAP-2', showgrid=True),
        margin=dict(l=40, r=20, t=60, b=40)
    )

    html_name = 'html/'+figure_name + '.html'
    fig.write_html(html_name)

    print('finished plotting, saved', html_name)


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

def umap_plot_2D_color_html(data,plot_samping=-1,figure_name='umap_plot_colored'):

    order_colors = np.arange(len(data))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data[:, 0],
        y=data[:, 1],
        mode='markers',
        marker=dict(
            size=6,
            color=order_colors,
            colorscale='Viridis',
            colorbar=dict(title='Order / Index')
        ),
        showlegend=False
    ))

    indices = np.arange(0, len(data), plot_samping)

    fig.add_trace(go.Scatter(
        x=data[indices, 0],
        y=data[indices, 1],
        mode='text',
        text=[str(i) for i in indices],
        textposition='top center',
        showlegend=False
    ))

    fig.update_layout(
        title="UMAP Visualization (Color = Order)",
        xaxis=dict(title='UMAP-1', showgrid=True),
        yaxis=dict(title='UMAP-2', showgrid=True),
        margin=dict(l=40, r=20, t=60, b=40)
    )

    html_name = 'html/'+figure_name + '.html'
    fig.write_html(html_name)

    print('finished UMAP plotting, saved', html_name)

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

def umap_plot_3D_html(data,labels=None,plot_samping=-1,figure_name='umap_plot_plot_3D'):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=labels,
            colorscale='Viridis',
            colorbar=dict(title='Label')
        ),
        showlegend=False
    ))

    indices = np.arange(0, len(data), plot_samping)

    fig.add_trace(go.Scatter3d(
        x=data[indices, 0],
        y=data[indices, 1],
        z=data[indices, 2],
        mode='text',
        text=[str(i) for i in indices],
        textposition='top center',
        showlegend=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='UMAP-1'),
            yaxis=dict(title='UMAP-2'),
            zaxis=dict(title='UMAP-3')
        ),
        title='UMAP 3D',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    html_name = 'html/'+figure_name + '.html'
    fig.write_html(html_name)

    print('finished UMAP 3D plotting, saved', html_name)



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

def umap_plot_3D_color_html(data,plot_samping=-1,figure_name='umap_plot_plot_3D_colored'):

    order_colors = np.arange(len(data))

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=order_colors,
            colorscale='Viridis',
            colorbar=dict(title='Order / Index')
        ),
        showlegend=False
    ))

    indices = np.arange(0, len(data), plot_samping)

    fig.add_trace(go.Scatter3d(
        x=data[indices, 0],
        y=data[indices, 1],
        z=data[indices, 2],
        mode='text',
        text=[str(i) for i in indices],
        textposition='top center',
        showlegend=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='UMAP-1', showgrid=True),
            yaxis=dict(title='UMAP-2', showgrid=True),
            zaxis=dict(title='UMAP-3', showgrid=True)
        ),
        title='UMAP 3D',
        margin=dict(l=0, r=0, t=40, b=0)
    )

    html_name = 'html/'+figure_name + '.html'
    fig.write_html(html_name)

    print('finished UMAP 3D plotting, saved', html_name)
