#!/usr/bin/bash

# input cpu
device=${1:-"gpu"}

conda create -n flowbench python=3.8 -y
# require 'source' instead of 'conda' in bash env.
source activate flowbench

# conda install env
if [ "$device" == "cpu" ]
then
  echo "Install the version with CPU only"
  conda install pytorch torchvision torchaudio cpuonly pyg tensorboard \
    matplotlib seaborn joblib networkx numba \
    ipykernel flake8 autopep8 graphviz jupyter ipywidgets pytest \
    -c pytorch -c pyg
elif [ "$device" == "gpu" ]
then
  echo "Install the version with CUDA"
  conda install pytorch torchvision torchaudio cudatoolkit=11.6 pyg tensorboard \
    matplotlib seaborn joblib networkx numba \
    ipykernel flake8 autopep8 graphviz jupyter ipywidgets pytest \
    graphviz pygraphviz \
    -c pytorch -c nvidia -c pyg
else
  echo "Please choose from 'cpu' or 'gpu'."
fi

# pip install additional packages
pip install deephyper ray pygod pyod class_resolver umap-learn combo
# problem with install pygraphviz
python setup.py develop
