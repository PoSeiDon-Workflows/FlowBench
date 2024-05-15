#!/usr/bin/env python3

import json
import os
import shutil
import pandas as pd
from zipfile import ZipFile

workflow_stats = {
    "1000genome": {"name": "1000 Genome", "stats": {}},
    "montage": {"name": "Montage", "stats": {}},
    "predict_future_sales": {"name": "Predict Future Sales", "stats": {}},
    "variant_calling": {"name": "Variant Calling", "stats": {}},
    "casa_wind_full": {"name": "CASA Wind Speed", "stats": {}},
    "casa_nowcast_full": {"name": "CASA Nowcast", "stats": {}},
    "casa_nowcast_small": {"name": "CASA Nowcast Small", "stats": {}},
    "somospie": {"name": "Soil Moisture", "stats": {}},
    "pycbc_inference": {"name": "PyCBC Inference", "stats": {}},
    #"pycbc_search": {"name": "PyCBC Search", "stats": {}},
    "eht_difmap": {"name": "EHT Difmap", "stats": {}},
    "eht_imaging": {"name": "EHT Imaging", "stats": {}},
    "eht_smili": {"name": "EHT Smili", "stats": {}}
}

def populateStats(all_stats):
    stats = {}
    stats["all_dags"] = all_stats["runID"].unique().size
    normal_dags = all_stats["runID"].unique().size

    anomaly_cnt = all_stats[all_stats["anomaly_type"] == "cpu_2"]["runID"].unique().size
    normal_dags -= anomaly_cnt
    stats["cpu_2_dags"] = anomaly_cnt

    anomaly_cnt = all_stats[all_stats["anomaly_type"] == "cpu_3"]["runID"].unique().size
    normal_dags -= anomaly_cnt
    stats["cpu_3_dags"] = anomaly_cnt

    anomaly_cnt = all_stats[all_stats["anomaly_type"] == "cpu_4"]["runID"].unique().size
    normal_dags -= anomaly_cnt
    stats["cpu_4_dags"] = anomaly_cnt

    anomaly_cnt = all_stats[all_stats["anomaly_type"] == "hdd_5"]["runID"].unique().size
    normal_dags -= anomaly_cnt
    stats["hdd_5_dags"] = anomaly_cnt

    anomaly_cnt = all_stats[all_stats["anomaly_type"] == "hdd_10"]["runID"].unique().size
    normal_dags -= anomaly_cnt
    stats["hdd_10_dags"] = anomaly_cnt

    stats["normal_dags"] = normal_dags

    stats["normal_nodes"] = all_stats["anomaly_type"].isnull().sum()
    stats["cpu_2_nodes"] = all_stats[all_stats["anomaly_type"] == "cpu_2"].shape[0]
    stats["cpu_3_nodes"] = all_stats[all_stats["anomaly_type"] == "cpu_3"].shape[0]
    stats["cpu_4_nodes"] = all_stats[all_stats["anomaly_type"] == "cpu_4"].shape[0]
    stats["hdd_5_nodes"] = all_stats[all_stats["anomaly_type"] == "hdd_5"].shape[0]
    stats["hdd_10_nodes"] = all_stats[all_stats["anomaly_type"] == "hdd_10"].shape[0]

    return stats

def printStatsTable(workflow_stats):
    str_builder = [
        '<table>',
        '<thead>',
        '<tr>',
            '<th rowspan="3">Workflow</th>',
            '<th colspan="2">DAG Information</th>',
            '<th colspan="6">#DAG Executions</th>',
            '<th colspan="6">#Total Nodes per Type</th>',
            '</tr>',
        '<tr>',
            '<th rowspan="2">Nodes</th>',
            '<th rowspan="2">Edges</th>',
            '<th rowspan="2">Normal</th>',
            '<th colspan="3">CPU</th>',
            '<th colspan="2">HDD</th>',
            '<th rowspan="2">Normal</th>',
            '<th colspan="3">CPU</th>',
            '<th colspan="2">HDD</th>',
        '</tr>',
        '<tr>',
        '<th>2</th>',
        '<th>3</th>',
        '<th>4</th>',
        '<th>5</th>',
        '<th>10</th>',
        '<th>2</th>',
        '<th>3</th>',
        '<th>4</th>',
        '<th>5</th>',
        '<th>10</th>',
        '</tr>',
        '</thead>',
        '<tbody>',
    ]

    for wf_name in workflow_stats:
        stats = workflow_stats[wf_name]["stats"]
        str_builder.append('<tr>')
        str_builder.append(f'<td>{workflow_stats[wf_name]["name"]}</td>')
        str_builder.append(f'<td>{stats["num_jobs"]}</td>')
        str_builder.append(f'<td>{stats["num_edges"]}</td>')
        str_builder.append(f'<td>{stats["normal_dags"] if stats["normal_dags"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["cpu_2_dags"] if stats["cpu_2_dags"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["cpu_3_dags"] if stats["cpu_3_dags"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["cpu_4_dags"] if stats["cpu_4_dags"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["hdd_5_dags"] if stats["hdd_5_dags"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["hdd_10_dags"] if stats["hdd_10_dags"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["normal_nodes"] if stats["normal_nodes"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["cpu_2_nodes"] if stats["cpu_2_nodes"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["cpu_3_nodes"] if stats["cpu_3_nodes"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["cpu_4_nodes"] if stats["cpu_4_nodes"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["hdd_5_nodes"] if stats["hdd_5_nodes"] > 0 else "-"}</td>')
        str_builder.append(f'<td>{stats["hdd_10_nodes"] if stats["hdd_10_nodes"] > 0 else "-"}</td>')
        str_builder.append('</tr>')
    
    str_builder.append('</tbody>')
    str_builder.append('</table>')
    
    print("\n".join(str_builder))


if __name__ == "__main__":

    for wf_name in workflow_stats:
        dag_file = f"adjacency_list_dags/{wf_name}.json"
        stats_zip = f"data/{wf_name}.zip"
        stats_folder = f"data/{wf_name}"
    
        with open(dag_file, 'r') as f:
            dag_obj = json.load(f)
    
        edges = 0
        for j in dag_obj:
            edges += len(dag_obj[j])
    
        all_stats = pd.DataFrame()
        
        with ZipFile(stats_zip, 'r') as f:
            os.mkdir(stats_folder)
            f.extractall(stats_folder)
    
    
        for f in os.listdir(stats_folder):
            run_number = f.split('-')[-1].split('.')[0]
    
            stats = pd.read_csv(os.path.join(stats_folder, f), sep=',')
            stats["runID"] = run_number
        
            all_stats = pd.concat([all_stats, stats])
    
        stats = populateStats(all_stats)
        stats["num_jobs"] = len(dag_obj)
        stats["num_edges"] = edges
    
        workflow_stats[wf_name]["stats"] = stats

        shutil.rmtree(stats_folder)

    printStatsTable(workflow_stats)
