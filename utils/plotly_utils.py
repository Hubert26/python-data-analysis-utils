# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:28:28 2024

@author: Hubert Szewczyk
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default='browser'
from pathlib import Path
import numpy as np

from utils.file_utils import create_directory

#%%
def create_subplots_plotly(n_plots, n_cols=2, figsize=(30, 5)):
    """
    Creates a figure with subplots in a grid layout.

    Parameters:
    - n_plots (int): The number of subplots to create.
    - n_cols (int, optional): The number of columns in the subplot grid. Default is 2.
    - figsize (tuple, optional): The size of the figure in inches (width, height). Default is (30, 5).
    
    Returns:
    - fig (plotly.graph_objects.Figure): The created Plotly figure object with subplots.
    """
    # Validate inputs
    if n_plots < 1:
        raise ValueError("The number of plots (n_plots) must be at least 1.")
    
    if n_cols < 1:
        raise ValueError("The number of columns (n_cols) must be at least 1.")

    # Determine the number of rows required to fit the plots
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create the figure with subplots
    fig = make_subplots(rows=n_rows, cols=n_cols)

    # Adjust the size of the figure
    fig.update_layout(height=figsize[1] * 100, width=figsize[0] * 100)

    return fig

#%%
def create_multi_series_scatter_plot_plotly(data, **kwargs):
    """
    Creates a scatter plot with multiple data series using Plotly.

    Parameters:
    - data: List of dictionaries, where each dictionary represents a dataset to plot.
            Each dictionary should have 'x' and 'y' keys for the data points.
    - kwargs: Additional keyword arguments for customization.

    Returns:
    - fig: Plotly figure with the created scatter plot.
    """
    # Extract additional keyword arguments
    legend_labels = kwargs.get('legend_labels', [])
    scatter_colors = kwargs.get('scatter_colors', [])
    plot_title = kwargs.get('plot_title', 'Multi-Series Scatter Plot')
    x_label = kwargs.get('x_label', 'X-axis')
    y_label = kwargs.get('y_label', 'Y-axis')
    show_grid = kwargs.get('show_grid', False)

    # Initialize the figure
    fig = go.Figure()

    # Add each series as a scatter trace
    for i, series in enumerate(data):
        name = legend_labels[i] if i < len(legend_labels) else f'Series {i+1}'
        color = scatter_colors[i] if i < len(scatter_colors) else None
        
        fig.add_trace(go.Scatter(
            x=series.get('x', []),
            y=series.get('y', []),
            mode='markers',
            name=name,
            marker=dict(color=color)
        ))

    # Update plot layout
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True,
        xaxis=dict(showgrid=show_grid),
        yaxis=dict(showgrid=show_grid)
    )

    return fig

#%%
def create_multi_series_histogram_plotly(data, **kwargs):
    """
    Creates a histogram with multiple data series using Plotly.

    Parameters:
    - data: List of dictionaries, where each dictionary represents a dataset to plot.
            Each dictionary should have a key 'x' for the data points.
    - kwargs: Additional keyword arguments for customization.

    Returns:
    - fig: Plotly figure with the created histogram plot.
    """
    # Extract additional keyword arguments
    legend_labels = kwargs.get('legend_labels', [])
    bar_colors = kwargs.get('bar_colors', [])
    plot_title = kwargs.get('plot_title', 'Multi-Series Histogram')
    x_label = kwargs.get('x_label', 'X-axis')
    y_label = kwargs.get('y_label', 'Count')
    show_grid = kwargs.get('show_grid', False)
    histnorm = kwargs.get('histnorm', None)  # 'percent', 'density', or None for counts

    # Initialize the figure
    fig = go.Figure()

    # Add each series as a histogram trace
    for i, series in enumerate(data):
        name = legend_labels[i] if i < len(legend_labels) else f'Series {i+1}'
        color = bar_colors[i] if i < len(bar_colors) else None
        
        fig.add_trace(go.Histogram(
            x=series.get('x', []),
            name=name,
            marker_color=color,
            opacity=0.75,
            histnorm=histnorm  # Normalization option, if provided
        ))

    # Update plot layout
    fig.update_layout(
        barmode='overlay',  # Overlay histograms
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True,
        xaxis=dict(showgrid=show_grid),
        yaxis=dict(showgrid=show_grid)
    )

    return fig

#%%
def save_html_plotly(fig, file_path: str) -> None:
    """
    Save a Plotly plot to a file in the specified format and directory.
    """
    supported_formats = ["html"]

    # Extract file extension
    file_extension = Path(file_path).suffix.lstrip('.')

    # Ensure valid format
    if file_extension not in supported_formats:
        raise ValueError(f"Unsupported format '{file_extension}'. Supported formats are: {', '.join(supported_formats)}.")

    # Ensure directory exists
    dir_path = Path(file_path).parent
    if dir_path.is_dir():
        create_directory(dir_path)

    # Check if fig is a Plotly figure
    if isinstance(fig, go.Figure):
        if file_extension == "html":
            pio.write_html(fig, file_path)
    else:
        raise TypeError("The 'fig' parameter must be a Plotly 'go.Figure'.")

#%%



#%%
def create_heatmap_plotly(data, **kwargs):
    """
    Creates a heatmap using Plotly with customizable parameters passed via kwargs.
    
    Parameters:
    - data: 2D list, numpy array, or pandas DataFrame representing heatmap values.
    - kwargs: Dictionary of keyword arguments for customization.
    
    Returns:
    - fig: Plotly Figure object containing the heatmap.
    """
    # Heatmap properties
    heatmap_props = kwargs.get("heatmap_props", {})
    heatmap_props = {
        "zmin": heatmap_props.get("zmin", None),  # Min value for color scale
        "zmax": heatmap_props.get("zmax", None),  # Max value for color scale
        "colorscale": heatmap_props.get("colorscale", "Viridis"),  # Color scale
        "reversescale": heatmap_props.get("reversescale", False),  # Reverse color scale
        "showscale": heatmap_props.get("showscale", True),  # Show color bar
    }
    
    # Axis labels
    axis_props = kwargs.get("axis_props", {})
    axis_props = {
        "x_title": axis_props.get("x_title", "X Axis"),
        "y_title": axis_props.get("y_title", "Y Axis"),
        "x_tickangle": axis_props.get("x_tickangle", 0),
        "x_tickvals": axis_props.get("x_tickvals", None),
        "x_ticktext": axis_props.get("x_ticktext", None),
        "y_tickvals": axis_props.get("y_tickvals", None),
        "y_ticktext": axis_props.get("y_ticktext", None),
    }
    
    # Title properties
    title_props = kwargs.get("title_props", {})
    title_props = {
        "text": title_props.get("text", "Heatmap"),
        "x": title_props.get("x", 0.5),
        "y": title_props.get("y", 0.9),
        "font_size": title_props.get("font_size", 16),
    }
    
    #Annotations
    annotation_props = kwargs.get("annotation_props", {})
    annotation_props = {
        "show_annotation": annotation_props.get("show_annotation", True),  # Show annotations
        "annotation_color": annotation_props.get("annotation_color", "auto")
        }

    # Grid properties
    grid_props = kwargs.get("grid_props", {})
    grid_props = {
        "show_grid": grid_props.get("show_grid", True),
        "grid_width": grid_props.get("grid_width", 0.5),
        "grid_color": grid_props.get("grid_color", "gray"),
    }
    
    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            zmin=heatmap_props["zmin"],
            zmax=heatmap_props["zmax"],
            colorscale=heatmap_props["colorscale"],
            reversescale=heatmap_props["reversescale"],
            showscale=heatmap_props["showscale"],
        )
    )
    
    # Update axis properties
    fig.update_layout(
        xaxis=dict(
            title=axis_props["x_title"],
            tickangle=axis_props["x_tickangle"],
            tickvals=axis_props["x_tickvals"],
            ticktext=axis_props["x_ticktext"],
            showgrid=grid_props["show_grid"],
            gridwidth=grid_props["grid_width"],
            gridcolor=grid_props["grid_color"],
        ),
        yaxis=dict(
            title=axis_props["y_title"],
            tickvals=axis_props["y_tickvals"],
            ticktext=axis_props["y_ticktext"],
            showgrid=grid_props["show_grid"],
            gridwidth=grid_props["grid_width"],
            gridcolor=grid_props["grid_color"],
        ),
    )
    
    if annotation_props['show_annotation']:
        annotations = []
        zmin = heatmap_props.get("zmin", np.min(data))
        zmax = heatmap_props.get("zmax", np.max(data))
        for i in range(len(data)):
            for j in range(len(data[0])):
                normalized_value = (data[i][j] - zmin) / (zmax - zmin) if zmax > zmin else 0.5
                
                if annotation_props['annotation_color'] == "auto":
                    text_color = "white" if normalized_value < 0.5 else "black"
                else:
                    text_color = annotation_props['annotation_color']
                    
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=str(data[i][j]),
                        showarrow=False,
                        font=dict(color=text_color)
                    )
                )
        fig.update_layout(annotations=annotations)
        
    # Update title
    fig.update_layout(
        title=dict(
            text=title_props["text"],
            x=title_props["x"],
            y=title_props["y"],
            font=dict(size=title_props["font_size"]),
        )
    )
    
    return fig

    
#%%
if __name__ == "__main__":
    current_working_directory = Path.cwd()
    output_file_path = current_working_directory / 'plots'


#%%
    data1 = np.random.randn(10, 10)
    data1 = np.round(data1, 2)
    
    heatmap1_fig = create_heatmap_plotly(
        data1,
        heatmap_props={"colorscale": "Cividis", "zmin": -1, "zmax": 1},
        axis_props={"x_title": "Columns", "y_title": "Rows", "x_tickangle": 45},
        annotation_props={"show_annotation": True},
        grid_props={"show_grid": True, "grid_color": "blue"},
        title_props={"text": "Custom Heatmap", "font_size": 20}
    )
       
    heatmap1_fig.show()
