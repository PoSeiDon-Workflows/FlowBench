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
    <td>51</td>
    <td>100</td>
    <td>25</td>
    <td>-</td>
    <td>100</td>
    <td>75</td>
    <td>32398</td>
    <td>5173</td>
    <td>756</td>
    <td>-</td>
    <td>5392</td>
    <td>4368</td>
  </tr>
  <tr>
    <td>Montage</td>
    <td>539</td>
    <td>2338</td>
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
</tbody>
</table>


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
│   ├── 1000genome.json
│   ├── montage.json
│   ├── predict_future_sales.json
│   └── README.md
├── images
│   ├── 1000genome-workflow.png
│   ├── chameleon_deployment.png
│   ├── model_comparison.png
│   ├── montage-workflow.png
│   ├── predict-future-sales-workflow.png
│   └── raw_data_to_parsed_data.png
├── LICENSE
├── parsed
│   ├── cpu_2
│   │   ├── 1000-genome-runXXXX.csv
│   │   ├── montage-runXXXX.csv
│   │   └── predict-future-sales-runXXXX.csv
│   ├── cpu_3
│   │   ├── 1000-genome-runXXXX.csv
│   │   ├── montage-runXXXX.csv
│   │   └── predict-future-sales-runXXXX.csv
│   ├── cpu_4
│   │   └── predict-future-sales-runXXXX.csv
│   ├── hdd_10
│   │   ├── 1000-genome-runXXXX.csv
│   │   ├── montage-runXXXX.csv
│   │   └── predict-future-sales-runXXXX.csv
│   ├── hdd_5
│   │   ├── 1000-genome-runXXXX.csv
│   │   ├── montage-runXXXX.csv
│   │   └── predict-future-sales-runXXXX.csv
│   ├── normal
│   │   ├── 1000-genome-runXXXX.csv
│   │   ├── montage-runXXXX.csv
│   │   └── predict-future-sales-runXXXX.csv
│   └── README.md
├── py_script
│   ├── benchmark
│   │   ├── pygod.py
│   │   └── pyod.py
│   ├── cartography.py
│   ├── dataset.py
│   ├── README.md
│   └── utils.py
├── raw
│   ├── archive
│   │   ├── elasticsearch.tar.xz.partXX
│   │   └── workflow-submit-dirs.tar.xz.partXX
│   ├── docker-compose.yml
│   ├── experiments.csv
│   ├── parse-data.py
│   ├── run-parser.sh
│   └── workflows.tar.xz
├── setup.py
├── setup.sh
└── README.md
```
