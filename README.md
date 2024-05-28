# Flow-Bench: A Dataset for Computational Workflow Anomaly Detection

Flow-Bench is a benchmark dataset for anomaly detection techniques in computational workflows.
Flow-Bench contains workflow execution traces, executed on distributed infrastructure, that include systematically injected anomalies (labeled), and offers both the raw execution logs and a more compact parsed version. 
In this GitHub repository, apart from the logs and traces, you will find sample code to load and process the parsed data using pytorch, as well as, the code used to parse the raw logs and events.

## Dataset

The dataset contains 1211 DAG executions from 2 computational science workflows and 1 ML data science workflow, under normal and anomalous conditions. These workflows were executed using [Pegasus WMS - Panorama](https://github.com/pegasus-isi/pegasus/tree/panorama). Synthetic anomalies, were injected using Docker’s runtime options to limit and shape the performance. The table bellow presents the breakdown of DAG executions per type, and the data have been labeled using 6 tags (normal, cpu_2, cpu_3, cpu_4, hdd_5 and hdd_10).

- *normal*: No anomaly is introduced - normal conditions.
- *CPU K*: M cores are advertised on the executor nodes, but on some nodes, K cores are not allowed to be used. (K = 2, 3, 4M = 4, 8 and K < M)
- *HDD K*: On some executor nodes, the average write speed to the disk is capped atK MB/s and the read speed at (2×K) MB/s. (K = 5, 10)

<table>
<thead>
<tr>
<th rowspan="3">Workflow</th>
<th colspan="2">DAG Information</th>
<th colspan="6">#DAG Executions</th>
<th colspan="6">#Total Nodes per Type</th>
</tr>
<tr>
<th rowspan="2">Nodes</th>
<th rowspan="2">Edges</th>
<th rowspan="2">Normal</th>
<th colspan="3">CPU</th>
<th colspan="2">HDD</th>
<th rowspan="2">Normal</th>
<th colspan="3">CPU</th>
<th colspan="2">HDD</th>
</tr>
<tr>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>10</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>10</th>
</tr>
</thead>
<tbody>
<tr>
<td>1000 Genome</td>
<td>137</td>
<td>289</td>
<td>50</td>
<td>100</td>
<td>25</td>
<td>-</td>
<td>100</td>
<td>75</td>
<td>32261</td>
<td>5173</td>
<td>756</td>
<td>-</td>
<td>5392</td>
<td>4368</td>
</tr>
<tr>
<td>Montage</td>
<td>539</td>
<td>2838</td>
<td>51</td>
<td>46</td>
<td>80</td>
<td>-</td>
<td>67</td>
<td>76</td>
<td>137229</td>
<td>4094</td>
<td>11161</td>
<td>-</td>
<td>8947</td>
<td>11049</td>
</tr>
<tr>
<td>Predict Future Sales</td>
<td>165</td>
<td>581</td>
<td>100</td>
<td>88</td>
<td>88</td>
<td>88</td>
<td>88</td>
<td>88</td>
<td>72609</td>
<td>3361</td>
<td>3323</td>
<td>3193</td>
<td>3321</td>
<td>3293</td>
</tr>
<tr>
<td>Variant Calling</td>
<td>371</td>
<td>792</td>
<td>80</td>
<td>80</td>
<td>80</td>
<td>-</td>
<td>75</td>
<td>80</td>
<td>115588</td>
<td>8287</td>
<td>7222</td>
<td>-</td>
<td>7365</td>
<td>8083</td>
</tr>
<tr>
<td>CASA Wind Speed</td>
<td>162</td>
<td>342</td>
<td>150</td>
<td>200</td>
<td>200</td>
<td>-</td>
<td>200</td>
<td>160</td>
<td>116836</td>
<td>8793</td>
<td>8382</td>
<td>-</td>
<td>8305</td>
<td>5104</td>
</tr>
<tr>
<td>CASA Nowcast</td>
<td>2081</td>
<td>4029</td>
<td>101</td>
<td>80</td>
<td>78</td>
<td>-</td>
<td>79</td>
<td>83</td>
<td>685045</td>
<td>49960</td>
<td>46664</td>
<td>-</td>
<td>46104</td>
<td>48328</td>
</tr>
<tr>
<td>CASA Nowcast Small</td>
<td>419</td>
<td>795</td>
<td>80</td>
<td>80</td>
<td>80</td>
<td>-</td>
<td>80</td>
<td>79</td>
<td>128562</td>
<td>10031</td>
<td>9592</td>
<td>-</td>
<td>9427</td>
<td>9569</td>
</tr>
<tr>
<td>Soil Moisture</td>
<td>60</td>
<td>185</td>
<td>125</td>
<td>98</td>
<td>97</td>
<td>-</td>
<td>92</td>
<td>93</td>
<td>24408</td>
<td>1706</td>
<td>1428</td>
<td>-</td>
<td>1344</td>
<td>1414</td>
</tr>
<tr>
<td>PyCBC Inference</td>
<td>17</td>
<td>26</td>
<td>206</td>
<td>89</td>
<td>74</td>
<td>-</td>
<td>67</td>
<td>66</td>
<td>6970</td>
<td>549</td>
<td>326</td>
<td>-</td>
<td>388</td>
<td>301</td>
</tr>
<tr>
<td>PyCBC Search</td>
<td>220</td>
<td>704</td>
<td>102</td>
<td>100</td>
<td>100</td>
<td>-</td>
<td>104</td>
<td>100</td>
<td>151004</td>
<td>9495</td>
<td>9039</td>
<td>-</td>
<td>8324</td>
<td>8346</td>
</tr>
<tr>
<td>EHT Difmap</td>
<td>33</td>
<td>59</td>
<td>142</td>
<td>93</td>
<td>88</td>
<td>-</td>
<td>89</td>
<td>89</td>
<td>13000</td>
<td>1059</td>
<td>737</td>
<td>-</td>
<td>877</td>
<td>860</td>
</tr>
<tr>
<td>EHT Imaging</td>
<td>12</td>
<td>18</td>
<td>212</td>
<td>76</td>
<td>74</td>
<td>-</td>
<td>70</td>
<td>69</td>
<td>4908</td>
<td>354</td>
<td>241</td>
<td>-</td>
<td>261</td>
<td>248</td>
</tr>
<tr>
<td>EHT Smili</td>
<td>16</td>
<td>26</td>
<td>148</td>
<td>87</td>
<td>84</td>
<td>-</td>
<td>90</td>
<td>93</td>
<td>6471</td>
<td>437</td>
<td>325</td>
<td>-</td>
<td>382</td>
<td>417</td>
</tr>
</tbody>
</table>

## Benchmark Installation

* Install the required packages by using `bash setup.sh`

## Benchmark Instructions

* load data as graphs in `pytorch_geometric` format:
  
  ```python
  from py_script.dataset import FlowDataset
  dataset = FlowDataset(root="./", name="montage")
  data = dataset[0]
  ```
  
  The `data` contains the structural information by accessing `data.edge_index`, and node feature information `data.x`.

* load data as tabular data in `pytorch` format:

  ```python
  from py_script.dataset import FlowDataset
  dataset = FlowDataset(root="./", name="montage")
  data = dataset[0]
  Xs = data.x
  ys = data.y
  ```

  Unlike the graph `data`, the `data` only contains the node features.

* load data as tabular data in `numpy` format:

  ```python
  from py_script.dataset import FlowDataset
  dataset = FlowDataset(root="./", name="montage")
  data = dataset[0]
  Xs = data.x.numpy()
  ys = data.y.numpy()
  ```

  This is the same as the previous one, but the data is in `numpy` format, which is typically used in the models from `sklearn` and `xgboost`.

## Benchmark Methods

* We provide benchmarks for anomaly detection based on [PyGOD](https://docs.pygod.org/en/latest/index.html) and [PyOD](https://pyod.readthedocs.io/en/latest/index.html) from graph data and tabular data, respectively.
* Checkout the script under `./py_script/benchmark/pygod.py` and `./py_script/benchmark/pyod.py` for more details.

## Benchmark Performance

<p align="center">
<img src="images/model_comparison.png" alt="Comparison of models using the benchmark dataset."/>
</p>

## Repository Structure

The repository is structured as follows:
- *adjacency_list_dags*: Contains json files of the executable workflow DAGs in adjacency list representation.
- *images*: Contains diagrams of the abstract workflow DAGs and of the processes used to generated the data.
- *parsed*: Contains the parsed version of the data. The folder is structured in subfolders per anomaly label.
- *py_script*: Contains scripts to load the dataset and run the benchmark.
- *raw*: Contains the raw logs and scripts to parse them.

```
.
├── adjacency_list_dags
├── benchmark
├── data 
│   ├── 1000genome.zip
│   ├── casa_nowcast_full.zip
│   ├── casa_wind_full.zip
│   ├── montage.zip
│   ├── predict_future_sales.zip
│   └── variant_calling.zip
├── examples
|   └── demo_xxx.py
├── flowbench
│   ├── nlp
│       ├── llm.py
│   ├── supervised
|   |   ├── mlp.py
|   |   ├── gnn.py
|   |   └── xxx.py
│   └── unsupervised
|       ├── gmm.py
|       ├── pca.py
|       └── xxx.py
├── flowbench.egg-info
├── hps/
├── tests/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
```
