# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:00:21 2024

@author: Hubert Szewczyk
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%%
def test_create_heatmap_default():
    """
    Test 1: Create a heatmap with default parameters.
    """
    from utils.matplotlib_utils import create_heatmap_matplotlib  # Import your function
    data = np.random.uniform(0, 1, size=(10, 10))
    ax = create_heatmap_matplotlib(
        data,
        axis_props={"x_title": "X-Axis", "y_title": "Y-Axis"},
        title_props={"text": "Default Heatmap"}
    )
    ax.set_title("Test 1: Default Parameters")
    plt.show()

#%%
def test_create_heatmap_restricted_colors():
    """
    Test 2: Create a heatmap with restricted color range and annotations.
    """
    from utils.matplotlib_utils import create_heatmap_matplotlib
    data = np.random.uniform(-5, 5, size=(15, 15))
    ax = create_heatmap_matplotlib(
        data,
        heatmap_props={"vmin": -3, "vmax": 3, "cmap": "coolwarm"},
        annotation_props={"show_annotation": True, "annotation_format": ".2f"},
        title_props={"text": "Restricted Color Range Heatmap"}
    )
    ax.set_title("Test 2: Restricted Color Range")
    plt.show()

#%%
def test_create_heatmap_large_data():
    """
    Test 3: Create a heatmap with a large data matrix.
    """
    from utils.matplotlib_utils import create_heatmap_matplotlib
    data = np.random.uniform(0, 100, size=(50, 50))
    ax = create_heatmap_matplotlib(
        data,
        heatmap_props={"cmap": "plasma"},
        title_props={"text": "Large Data Heatmap"},
        grid_props={"show_grid": False},
    )
    ax.set_title("Test 3: Large Data Matrix")
    plt.show()

#%%
def test_create_heatmap_negative_values_no_annotations():
    """
    Test 4: Create a heatmap with negative values and no annotations.
    """
    from utils.matplotlib_utils import create_heatmap_matplotlib
    data = np.random.uniform(-10, 10, size=(20, 20))
    ax = create_heatmap_matplotlib(
        data,
        heatmap_props={"cmap": "seismic", "vmin": -10, "vmax": 10},
        annotation_props={"show_annotation": False},
        title_props={"text": "Negative Values Heatmap"}
    )
    ax.set_title("Test 4: Negative Values without Annotations")
    plt.show()
    
#%%








#%%
if __name__ == "__main__":
    current_working_directory = Path.cwd()
    output_file_path = current_working_directory / 'plots'
    plt.close('all')

#%%
    #Tests for function create_heatmap_matplotlib
    
    test_create_heatmap_default()
    test_create_heatmap_restricted_colors()
    test_create_heatmap_large_data()
    test_create_heatmap_negative_values_no_annotations()