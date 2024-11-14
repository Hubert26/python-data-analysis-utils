# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:05:42 2024

@author: Hubert26
"""

from setuptools import setup, find_packages

setup(
    # Name of your project
    name="python-utils", 
    
    # Version of your project
    version="0.1",  
    
    # This will find all the packages within the 'utils' folder (i.e. all the .py files)
    packages=find_packages(where='utils'),
    
    # General dependencies that are required for the whole project
    install_requires=[
        "numpy",  # e.g., if numpy is used across the utils
        "pandas", # if pandas is needed in general utils
    ],
    
    # Extra dependencies for specific tools, defined under 'extras_require'
    extras_require={
        "math": [
            "scipy",  # Dependency for mathematical utilities
        ],
        "plotly": [
            "plotly",  # Dependency for Plotly-based visualizations
        ],
        "matplotlib": [
            "matplotlib",  # Dependency for Matplotlib-based visualizations
        ],
        "ml": [
            "scikit-learn",  # Dependency for machine learning tools (if you have ML-based functions)
        ],
        # The 'full' option will install all dependencies
        "full": [  
            "scipy",         # For math tools
            "plotly",        # For Plotly visualizations
            "matplotlib",    # For Matplotlib visualizations
            "scikit-learn",  # For ML tools
        ],
    },
)
