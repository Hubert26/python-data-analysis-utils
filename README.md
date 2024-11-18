# python-utils
A collection of versatile utility functions for data processing and analysis in Python. This library includes helper modules for handling data frames, file operations, mathematical computations, signal processing, data visualization, and more. Designed to streamline repetitive tasks and enhance code reusability across projects. 

## Install python-utils as a Dependency in `your_project`
To use the python-utils library in your project, you can install it locally using pip. The package includes optional dependencies for specific functionalities such as plotting, data processing, and signal analysis. Follow the steps below for installation:

### 1. Clone the repository:
Clone the `python-utils` repository to your preferred location::
```
git clone https://github.com/Hubert26/python-utils.git
```

### 3. Install the `python-utils` library with the desired optional dependencies:
Specify the path to the python-utils directory during installation. Choose the appropriate installation option:
+ Full Installation (includes all optional dependencies):
```
pip install "/path/to/python-utils[full]"
```
+ For specific functionalities:
```
pip install "/path/to/python-utils[plotly,matplotlib,dataframe,math_signals]"
```
+ Core Installation (without extras):
```
pip install "/path/to/python-utils"
```
### Key Notes:
1. Replace `"/path/to/python-utils"` with the actual path to the `python-utils` directory. This can be:
+ An absolute path, e.g., `C:/repositories/python-utils`.
+ A relative path from your current working directory, e.g., `../python-utils`.
2. Any changes to the `python-utils` project will reflect automatically in this setup because of the editable installation.

## Using python-utils as a Standalone Project
### 1. Clone the Repository:
```
git clone https://github.com/Hubert26/python-utils.git
```

### 2. Navigate to the python-utils project directory:
```
cd python-utils
```

### 3. Create the Conda environment from the file:
```
conda env create -f environment.yml
```
### 4. Activate the environment:
```
conda activate python-utils-env
```

## Requirements
+ Python >= 3.8
+ Dependencies for optional features are listed in the optional-dependencies section of `pyproject.toml`.

