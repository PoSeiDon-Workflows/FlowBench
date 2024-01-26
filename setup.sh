#!/usr/bin/bash

# check cuda version
if module spider cuda &> /dev/null 
then
    module load cuda
    CUDA=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c 1-4)
    PYG_CUDA=$(echo $CUDA | tr -d .)
    echo "CUDA is available, install the package with CUDA $CUDA"

    conda create -n flowbench python=3.10 -y
    source activate flowbench
    conda install pytorch=2.0.0 torchvision torchaudio pytorch-cuda=$CUDA pyg \
        matplotlib seaborn networkx numba \
        ipykernel flake8 autopep8 graphviz pygraphviz jupyter ipywidgets pytest \
        -c pytorch -c nvidia -c pyg -y
    # pip install tensorflow[and-cuda] # install tensorflow with cuda
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.0.0+cu${PYG_CUDA}.html
else
    echo "CUDA is not available, install the package with CPU only"
    conda create -n flowbench python=3.10 -y
    source activate flowbench
    conda install pytorch=2.0.0 torchvision torchaudio cpuonly pyg \
        matplotlib seaborn networkx numba \
        ipykernel flake8 autopep8 graphviz pygraphviz jupyter ipywidgets pytest \
        -c pytorch -c pyg -y
    # pip install tensorflow # install tensorflow without cuda
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
fi
# install optional packages
pip install lightning tensorboard deepspeed deephyper ray pygod pyod class_resolver \
    umap-learn combo scikit-learn-intelex -U -q
# install current package in develop mode
pip install -e .