# Flow-Bench

## Dataset
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

## Repository Structure
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
├── move_raw_data.sh
├── parsed
│   ├── cpu_2
│   ├── cpu_3
│   ├── cpu_4
│   ├── hdd_10
│   ├── hdd_5
│   ├── normal
│   └── README.md
├── raw
│   └── raw_data.xz
├── raw-temp
│   ├── docker-compose.yml
│   ├── elasticsearch
│   ├── experiments-all.csv
│   ├── experiments.csv
│   ├── extend-log.py
│   ├── parse-data.py
│   ├── workflows
│   └── workflow-submit-dirs
└── README.md
```

## 1000 Genome Workflow
![Alt text](/images/1000genome-workflow.png "1000 Genome Workflow")

## Montage Workflow
![Alt text](/images/montage-workflow.png "Montage Workflow")

## Predict Future Sales Workflow
![Alt text](/images/predict-future-sales-workflow.png "Predict Future Sales Workflow")
