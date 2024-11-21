# python-utils
A collection of versatile utility functions for data processing and analysis in Python. This library includes helper modules for handling data frames, file operations, mathematical computations, signal processing, data visualization, and more. Designed to streamline repetitive tasks and enhance code reusability across projects. 

##Installation

### Install python-utils as a Dependency in `your_project`
To use the python-utils library in your project, you can install it locally using pip. The package includes optional dependencies for specific functionalities such as plotting, data processing, and signal analysis. Follow the steps below for installation:

#### 1. Clone the repository:
Clone the `python-utils` repository to your preferred location::
```
git clone https://github.com/Hubert26/python-utils.git
```

#### 2. Navigate to your project directory:
Go to the directory of your project (`your_project`) where you want to install `python-utils` as a dependency:
```
cd /path/to/your_project
```

#### 3. Install the `python-utils` library with the desired optional dependencies:
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
#### Key Notes:
1. Replace `"/path/to/python-utils"` with the actual path to the `python-utils` directory. This can be:
+ An absolute path, e.g., `C:/repositories/python-utils`.
+ A relative path from your current working directory, e.g., `../python-utils`.
2. Any changes to the `python-utils` project will reflect automatically in this setup because of the editable installation.
3. You can directly import modules from the `utils` package without referencing `src`. This is configured in the `pyproject.toml` file using the `[tool.setuptools]` section.
Example Usage in Your Project
```
# Importing a specific module
from utils.plotly_utils import create_scatter_plot

# Importing multiple utilities
from utils import file_utils, math_utils
```

### Using python-utils as a Standalone Project
### Using `environment.yml` with Conda
If you prefer using Conda to set up the environment, follow these steps:
1. Clone the Repository:
```
git clone https://github.com/Hubert26/python-utils.git
```

2. Navigate to the `python-utils` project directory:
```
cd "/path/to/python-utils"
```

3. Create the Conda environment from the `environment.yml` file:
```
conda env create -f environment.yml
```
4. Activate the environment:
```
conda activate python-utils-env
```

#### Using `requirements.txt` with pip
If you prefer using `pip` and `requirements.txt` for installation, follow these steps:
1. Clone the Repository:
```
git clone https://github.com/Hubert26/python-utils.git
```

2. Navigate to the `python-utils` project directory:
```
cd "/path/to/python-utils"
```

3. Install the required dependencies using `pip`:
```
pip install -r requirements.txt
```

#### Key Notes:
1. Replace `"/path/to/python-utils"` with the actual path to the `python-utils` directory. This can be:
+ An absolute path, e.g., `C:/repositories/python-utils`.
+ A relative path from your current working directory, e.g., `../python-utils`.

### Requirements
+ Python >= 3.8
+ Dependencies for optional features are listed in the optional-dependencies section of `pyproject.toml`.

## Code Structure
The `python-utils` project is organized into a modular structure to ensure clarity, maintainability, and scalability. Below is an overview of the directory and file structure:
```
python-utils/
│
├── src/                  # Main source code directory
│   ├── utils/            # Utility modules
│   │   ├── __init__.py   # Package initializer
│   │   ├── dataframe_utils.py    # Utilities for DataFrame manipulation
│   │   ├── file_utils.py         # File operations helpers
│   │   ├── math_utils.py         # Mathematical computation utilities
│   │   ├── signal_utils.py       # Signal processing functions
│   │   ├── matplotlib_utils.py   # Matplotlib-based plotting utilities
│   │   ├── plotly_utils.py       # Plotly-based plotting utilities
│   │   └── string_utils.py       # String manipulation functions
│   │
│   ├── config.py          # Configuration settings for the project
│
├── tests/                 # Unit and integration tests
│   ├── __init__.py        # Test package initializer
│   ├── test_*.py          # Test modules for respective utilities
│
├── environment.yml        # Conda environment definition file
├── pyproject.toml         # Project metadata and dependencies (PEP 621)
├── requirements.txt       # Alternative dependency file for pip users
├── README.md              # Project documentation
├── LICENSE                # License information
└── .gitignore             # Git ignore rules
```

### Key Directories and Components:
+ `src/utils/`: Contains modular utility functions organized by category (e.g., DataFrame operations, plotting, signal processing). Each module is designed for a specific set of tasks.
+ `tests/`: Includes test cases for validating the functionality of the utility functions and modules.
+ `config.py`: Centralized configuration file for managing constants and settings used across the library.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Hubert26/python-utils/blob/main/LICENSE) file for details.
