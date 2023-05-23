# README

This is the python interface to read the workflow data. 

## Installation

* Install the required packages as `bash setup.sh`

## Instruction

* load data as graphs in `pytorch_geometric` format:
  
  ```python
  from py_script import PoSeiDon
  dataset = PoSeiDon(root="./", name="1000genome")
  data = dataset[0]
  ```
  
  The `data` contains the structural information by accessing `data.edge_index`, and node feature information `data.x`.

* load data as tabular data in `pytorch` format:

  ```python
  from py_script import PoSeiDon
  dataset = PoSeiDon(root="./", name="1000genome")
  data = dataset[0]
  Xs = data.x
  ys = data.y
  ```

  Unlike the graph `data`, the `data` only contains the node features.

* load data as tabular data in `numpy` format:

  ```python
  from py_script import PoSeiDon
  dataset = PoSeiDon(root="./", name="1000genome")
  data = dataset[0]
  Xs = data.x.numpy()
  ys = data.y.numpy()
  ```

  This is the same as the previous one, but the data is in `numpy` format, which is typically used in the models from `sklearn` and `xgboost`.

## Benchmark Methods

* We provide benchmarks for anomaly detection based on [PyGOD](https://docs.pygod.org/en/latest/index.html) and [PyOD](https://pyod.readthedocs.io/en/latest/index.html) from graph data and tabular data, respectively.
* Checkout the script under `./py_script/benchmark/pygod.py` and `./py_script/benchmark/pyod.py` for more details.

## Data Analytics

* [x] Feature processing
* [x] Data analysis
* Data visualization
* [x] Data cleaning
