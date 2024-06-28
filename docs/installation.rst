Installation
=========

FlowBench is a Python package that provides a set of tools for the analysis workflow logs. It is supported to have `scikit-learn <https://scikit-learn.org/stable/>`_ and `PyTorch <https://pytorch.org/>`_ as its backend, as well as `PyTorch-Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ for graph-based analysis. 

To install this project, we will create a virtual env in conda following these steps:

1. Clone the repository:

    ```
    git clone https://github.com/PoSeiDon-Workflows/FlowBench.git
    ```

2. Navigate into the project directory, install the package
  
    ```
    cd FlowBench
    bash setup.sh
    ```
    
  It will create a virtual environment `flowbench` and install all the required dependencies.

3. Activate the virtual environment

    ```
    conda activate flowbench
    ```

4. Run the demo with GMM API from PyOD
  
    ```
    python example/demo_pyod.py
    ```
