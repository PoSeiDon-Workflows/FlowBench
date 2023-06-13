# Flow-Bench: A Dataset for Computational Workflow Anomaly Detection

Flow-Bench is a benchmark dataset for anomaly detection techniques in computational workflows.
Flow-Bench contains workflow execution traces, executed on distributed infrastructure, that include systematically injected anomalies (labeled), and offers both the raw execution logs and a more compact parsed version. 
In this GitHub repository, apart from the logs and traces, you will find sample code to load and process the parsed data using pytorch, as well as, the code used to parse the raw logs and events.

## Workflows
### 1000 Genome Workflow

The 1000 genome project provides a reference for human variation, having reconstructed the genomes of 2,504 individuals across 26 different populations. The test case we have here, identifies mutational overlaps using data from the 1000 genomes project in order to provide a null distribution for rigorous statistical evaluation of potential disease-related mutations. The implementation of the worklfow can be found here: https://github.com/pegasus-isi/1000genome-workflow.

![Alt text](/images/1000genome-workflow.png "1000 Genome Workflow")

### Montage Workflow

Montage is an astronomical image toolkit with components for re-projection, background matching, co-addition, and visualization of FITS files. Montage workflows typically follow a predictable structure based on the inputs, with each stage of the workflow often taking place in discrete levels separated by some synchronization/reduction tasks. The implementation of the workflow can be found here:  https://github.com/pegasus-isi/montage-workflow-v3.

![Alt text](/images/montage-workflow.png "Montage Workflow")

### Predict Future Sales Workflow

The predict future sales workflow provides a solution to Kaggle’s predict future sales competition. The workflow receives daily historical sales data from January 2013 to October 2015 and attempts to predict the sales for November 2015. The workflow includes multiple preprocessing and feature engineering steps to augment the dataset with new features and separates the dataset into three major groups based on their type and sales performance. To improve the prediction score, the workflow goes through a hyperparameter tuning phase and trains 3 discrete XGBoost models for each item group. In the end, it applies a simple ensemble technique that uses the appropriate model for each item prediction and combines the results into a single output file. The implementation of the workflow can be found here: https://github.com/pegasus-isi/predict-future-sales-workflow.

![Alt text](/images/predict-future-sales-workflow.png "Predict Future Sales Workflow")


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

![Alt text](/images/model_comparison.png "Comparison of models using the benchmark dataset.")


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
│   └── predict_future_sales.json
├── images
│   ├── 1000genome-workflow.png
│   ├── chameleon_deployment.png
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
