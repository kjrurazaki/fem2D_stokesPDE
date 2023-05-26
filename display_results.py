import numpy as np

from matplotlib import tri as mtri
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LinearLocator
from scipy.ndimage import laplace
from mpl_toolkits.mplot3d import Axes3D

def plot_field_2D(coord,
               triang,
               nodal_values,
               nodal = True,
               ax = None,
               label_cbar = None,
               cmap = 'viridis'
            ):
    """
    Plot color mapping in 2D geometry
    """
    colors = nodal_values.copy()
    triangulation = mtri.Triangulation(coord[:, 0], 
                                       coord[:, 1], 
                                       triang[:, :] - 1)
    if ax == None:
        fig = plt.figure()
        ax = fig.gca()
    if nodal == True:
        c = ax.tricontourf(triangulation, 
                           colors,
                           cmap = cmap)
        suffix = "_nodal"
    else:
        c = ax.tripcolor(triangulation, colors, 
                         shading="gouraud", 
                         antialiased=True,
                         cmap = cmap)
        suffix = "_triang"
    cbar = plt.colorbar(c, shrink = 1, format="%.2e", ax = ax)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0 ,1])
    ax.set_aspect(1)

def plot_field_3D(coord,
            triang,
            nodal_values,
            nodal=True,
            limit_z = False,
            figsize = [15,5],
            color_map = False
        ):
        """
        Plot mapping in 3D geometry
        """
        plt.style.use('classic')
        triangulation = mtri.Triangulation(coord[:, 0], 
                                            coord[:, 1], 
                                            triang[:, :] - 1)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        if nodal == True:
            c = nodal_values
        else:
            c = nodal_values[triangulation.triangles].mean(axis=1)

        def plot_subfield(ax, elev, azim, limit_z):
            if color_map == True:
                ax.plot_trisurf(triangulation, nodal_values, linewidth = 0, shade = False, cmap = plt.cm.coolwarm)
            elif color_map == False:
                ax.plot_trisurf(triangulation, nodal_values, shade=False, color = 'Gray')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim([0, 1])
            ax.set_ylim([0 ,1])
            if limit_z == True:
                ax.set_zlim([0 ,1])
            ax.view_init(elev=elev, azim=azim)
            ax.xaxis.set_major_locator(LinearLocator(numticks=2))
            ax.yaxis.set_major_locator(LinearLocator(numticks=2))
            ax.zaxis.set_major_locator(LinearLocator(numticks=2))

        # plot the 3D scatter plot with different angles
        plot_subfield(ax1, elev=10, azim= -45, limit_z = limit_z)
        plot_subfield(ax2, elev=45, azim= - 90, limit_z = limit_z)
        plot_subfield(ax3, elev=0, azim= - 45, limit_z = limit_z)

        if nodal == True:
            suffix = "_nodal"
        else:
            suffix = "_triang"
        fig.set_facecolor('white')

def plot_laplacian(grid, x_unique, y_unique, ax = None):
    """
    Laplacian (Image filter)
    Plot and return the computed grid
    Reference: https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm#:~:text=The%20Laplacian%20is%20a%202,see%20zero%20crossing%20edge%20detectors).
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.gca()
    laplacian = laplace(grid)
    plot = ax.imshow(laplacian, cmap='viridis', origin='lower')
    plt.colorbar(plot, shrink = 0.5, label='Laplacian', ax = ax)
    plt.xticks(range(grid.shape[1]), x_unique)
    plt.yticks(range(grid.shape[0]), y_unique)
    return laplacian

def save_fig(name):
    """
    Save the figure
    """
    plt.savefig(
        f"figures/fig_{name}.pdf",
        bbox_inches="tight",
        format="pdf",
    )

def dataframe_to_grid(df):
    """
    Transform a dataframe with columns x, y and value to a grid
    The final matrix has the values inserted in the position x, y
    If x, y are flots the code transforms it by enumerating
    """
    x_unique = sorted(df['x'].unique())
    y_unique = sorted(df['y'].unique())
    
    x_index_map = {x: i for i, x in enumerate(x_unique)}
    y_index_map = {y: i for i, y in enumerate(y_unique)}

    grid = np.empty((len(y_unique), len(x_unique)))
    for _, row in df.iterrows():
        x, y, concentration = row['x'], row['y'], row['concentration']
        grid[y_index_map[y], x_index_map[x]] = concentration

    return grid

def plot_derivative_grid(grid, x_unique, y_unique, min_line_width=1, max_line_width=5):
    """
    Plot derivatives of the grid generated by dataframe_to_grid as lines
    Color of line indicate if the derivative is positive or negative (upward)
    Width of the line gives the magnitude
    ---
    Call example:
        # Your grid data
        grid = dataframe_to_grid(df_concentration)
        x_unique = sorted(df_concentration['x'].unique())
        y_unique = sorted(df_concentration['y'].unique())
        plot_grid(grid, x_unique, y_unique, min_line_width=1, max_line_width=20)
    """
    plt.figure(figsize=(grid.shape[1], grid.shape[0]))

    max_change = np.max(np.abs(np.diff(grid, axis=0))) + np.max(np.abs(np.diff(grid, axis=1)))

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            #plt.plot(x, y, marker='o', markersize=10, label=f"({x}, {y})")
            
            if y < grid.shape[0] - 1:
                change = np.abs(grid[y+1, x] - grid[y, x])
                normalized_change = change / max_change
                line_width = min_line_width + normalized_change * (max_line_width - min_line_width)
                color = 'b' if grid[y+1, x] > grid[y, x] else 'r'
                plt.plot([x, x], [y, y+1], color=color, linewidth=line_width)
            
            if x < grid.shape[1] - 1:
                change = np.abs(grid[y, x+1] - grid[y, x])
                normalized_change = change / max_change
                line_width = min_line_width + normalized_change * (max_line_width - min_line_width)
                color = 'b' if grid[y, x+1] > grid[y, x] else 'r'
                plt.plot([x, x+1], [y, y], color=color, linewidth=line_width)

    plt.xticks(range(grid.shape[1]), x_unique)
    plt.yticks(range(grid.shape[0]), y_unique)
    plt.gca().invert_yaxis()
    plt.grid(True)  
    plt.show()

def plot_residuals(df, method, x_scale = ['log', 'log'], y_scale = ['log', 'linear'], c_color = 'diff', name = None):
    """
    The scatter plot is color-coded based on the categorical variable diff by default. 
    """
    xaxis_dictmethod = {
    'penalty' : ['penalty', 'log(Penalty)'],
    'lifting' : ['id', 'ID']
    }
    # Filter method
    df_plot = df[df['method'] == method]
    
    # Residuals using penalty method
    fig, ax = plt.subplots(1, 2, figsize=[20, 8])

    # create a custom colormap with colors based on number of unique values in c_color column
    unique_vals = df.loc[df['method'] == method, c_color].unique()
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(unique_vals)]
    cmap = mcolors.ListedColormap(colors)
           
    # Color encode the c_color column
    encoded = df.loc[df['method'] == method, c_color].astype('category').cat.codes
        
    def plot_scatter(df, ax, loc_ax, y_axis):
        yaxis_dict = {'residual_system': 'System residual',
                     'residual_bc': 'Boundary condition residual'}
        # plot the first scatter plot
        scatter = ax.scatter(df[xaxis_dictmethod[method][0]], 
                   df[y_axis],
                   c = encoded,
                   cmap = cmap,
                   s = 40)
        
        ax.set_yscale(y_scale[loc_ax])
        ax.set_xscale(x_scale[loc_ax])
        ax.set_ylabel(f'log({yaxis_dict[y_axis]})', fontsize = 14)
        ax.set_xlabel(xaxis_dictmethod[method][1], fontsize = 14)
        return scatter
    
    plot_scatter(df_plot, ax[0], 0, 'residual_system')
    scatter = plot_scatter(df_plot, ax[1], 1, 'residual_bc')
    
    # create a color bar with color indicators in the middle of each color
    bounds = np.linspace(0, len(unique_vals), len(unique_vals) + 1)
    ticks = np.linspace(0.5, len(unique_vals) - 0.5, len(unique_vals))
    cbar = plt.colorbar(scatter, boundaries=bounds, ticks=ticks)
    cbar.ax.set_yticklabels([f'{val:.2f}' for val in unique_vals])
    cbar.set_label('Diffusion coefficient', fontsize = 14)
    save_fig(f'{name}_residuals')
    plt.show()


def plot_id(df, nodes, triangles, id):
    """
    Plot the 3D of the concentration
    """
    df = df.loc[df['id'] == id].copy()
    # Setting the nodes
    df['x'],  df['y'] = nodes[:, 0], nodes[:, 1]
    plot_field_3D(nodes,
                  triangles,
                  df['concentration'],
                  nodal = True,
                  limit_z = False,
                  figsize = [25, 10])
    save_fig(f'{id}_3Dplot')
    plt.show()

def plot_idprofile(df, nodes, triangles, id):
    """
    Plot profiles (concentration and laplacian)
    """
    df = df.loc[df['id'] == id].copy()
    # Setting the nodes
    df['x'],  df['y'] = nodes[:, 0], nodes[:, 1]

    fig, ax = plt.subplots(1, 2, figsize =[20, 15])
    plot_field_2D(nodes, triangles, df['concentration'], ax = ax[0], label_cbar = 'Concentration')

    grid = dataframe_to_grid(df)
    x_unique = sorted(df['x'].unique())
    y_unique = sorted(df['y'].unique())
    laplacian = plot_laplacian(grid, x_unique, y_unique, ax[1])
    print(f'Total absolute derivatives: {sum(sum(np.absolute(laplacian)))}')
    print(f'Mean absolute derivatives: {np.mean(np.absolute(laplacian))}')

    fig.autofmt_xdate()
    fig.set_facecolor('white')
    save_fig(f'{id}_2Dprofile')
    plt.show()


def metric_laplacian(df, nodes, id):
    """
    Compute the total sum and mean of laplacian for the id case
    """
    df = df.loc[df['id'] == id].copy()
    # Setting the nodes
    df['x'],  df['y'] = nodes[:, 0], nodes[:, 1]
    grid = dataframe_to_grid(df)
    laplacian = laplace(grid)
    return [sum(sum(np.absolute(laplacian))), np.mean(np.absolute(laplacian))]

def plot_geometry(
    coord,
    triang,
    show_vertices=True,
    show_elements=True,
    figsize = [20, 20],
    save_name = None,
    color_elements = 'k',
    linewidth_elements = 0.2,
    ax = None
):
    """
    Plot geometry with or without elements
    """
    if ax == None:
        fig = plt.figure(figsize = figsize)
        ax = fig.gca()
    if show_vertices == True:
        ax.plot(coord[:, 0], coord[:, 1], ".")

    if show_elements == True:
        ax.triplot(
            coord[:, 0],
            coord[:, 1],
            triang[:, :] - 1,
            color = color_elements,
            linewidth = linewidth_elements,
        )
    if ax == None:
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.set_aspect(1)
        if save_name == None:
            plt.savefig(
                f"figures/fig_geo.pdf", bbox_inches="tight", format="pdf"
            )
        else:
            plt.savefig(
                f"figures/{save_name}.pdf", bbox_inches="tight", format="pdf"
            )