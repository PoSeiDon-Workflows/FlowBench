import glob
import os
import os.path as osp
import shutil
import zipfile

import pandas as pd

from flowbench import list_workflows
from flowbench.utils import create_dir, parse_adj

TS_FEATURES = ["ready",
               "submit",
               "execute_start",
               "execute_end",
               "post_script_start",
               "post_script_end",
               "stage_in_start",
               "stage_in_end"]
DELAY_FEATURES = ["wms_delay",
                  "queue_delay",
                  "runtime",
                  "post_script_delay",
                  "stage_in_delay",
                  "stage_out_delay"]
BYTES_FEATURES = ["stage_in_bytes",
                  "stage_out_bytes"]
KICKSTART_FEATURES = ["kickstart_executables_cpu_time"]


def load_tabular_data(name="1000genome",
                      raw_folder="data",
                      columns=None,
                      binary=True):
    r""" Load the tabular data from `raw_data` folder.

    Args:
        name (str, optional): Name of the workflow. Defaults to "1000genome".
        raw_folder (str, optional): The folder to load the data. Defaults to "data".
        columns (list, optional): Columns of features to be select.
            Defaults to None, to select all the features.
        binary (bool, optional): Whether to convert the target to binary. Defaults to True.

    Returns:
        pd.DataFrame: A dataframe of combined data
    """
    # extract zip file
    _folder = osp.join("/tmp", "zipfile", name)

    # Check if _folder exists and remove it
    if os.path.exists(_folder):
        shutil.rmtree(_folder)
    with zipfile.ZipFile(f"{raw_folder}/{name}.zip", 'r') as f:
        f.extractall(_folder)

    files = glob.glob(f"{_folder}/*.csv")
    df_list = []
    for file in files:
        df = pd.read_csv(file, index_col=[0])
        nodes, edges = parse_adj(name)
        # change the index the same as `nodes`
        for i, node in enumerate(df.index.values):
            if node.startswith("create_dir_") or node.startswith("cleanup_"):
                new_name = node.split("-")[0]
                df.index.values[i] = new_name

        # sort node name in json matches with node in csv.
        df = df.iloc[df.index.map(nodes).argsort()]
        # df.index = df.index.map(nodes)

        # subtract the timestamp by the first timestamp (ready)
        df[TS_FEATURES] = df[TS_FEATURES].sub(df[TS_FEATURES].ready.min())

        df = df.fillna(0)
        df_list.append(df)
        os.remove(file)

    # concatenate list of dataframes
    merged_df = pd.concat(df_list)

    # select features
    if columns is None:
        selected_features = merged_df.columns.to_list()
    else:
        if isinstance(columns, str):
            selected_features = [columns]
        else:
            selected_features = columns

    # add `label`
    if binary:
        merged_df['label'] = merged_df["anomaly_type"].map(lambda x: 0 if x == 0 else 1)
        merged_df = merged_df[selected_features + ['label']]
    else:
        _multi_labels = list(merged_df["anomaly_type"].unique())
        _multi_cat = [cat.split("_")[0] for cat in _multi_labels if cat != "None"]
        label_map = {label: i + 1 for i, label in enumerate(_multi_cat) if label != "None"}
        label_map["None"] = 0
        merged_df['label'] = merged_df["anomaly_type"].map(label_map)
        merged_df = merged_df[selected_features + ['label']]

    return merged_df


def build_text_data(df,
                    folder="./",
                    name="1000genome",
                    **kwargs):
    """ Convert the tabular data into text data with columns of ['text', 'label']
        "<COLUMN> is <VALUE> <COLUMN> is <VALUE> ... ,<LABEL>"

    Args:
        df (pd.DataFrame): Dataframe of concated data.
        folder (str, optional): Folder name to be processed. Defaults to "./".
        name (str, optional): Name of the workflow. Defaults to "1000genome".

    Returns:
        str: File name of the output csv file.
    """
    output_dir = osp.join(folder, name)
    create_dir(output_dir)
    outfile = osp.join(output_dir, "all.csv")
    with open(outfile, "w") as f:
        f.write("text,label\n")
        for index, row in df.iterrows():
            row_str = ""
            for col in df.columns:
                if col != "label":
                    # row_str += f"{col} is {row[col]} "
                    row_str += f"{' '.join(col.split('_'))} is {str(row[col]).replace(',', ' ')} "
            row_str += f",{int(row['label'])}"
            row_str += "\n"
            f.write(row_str)

    return outfile


if __name__ == "__main__":
    data_folder = "./data"
    for wf in list_workflows()[:]:
        print("processing", wf)
        df = load_tabular_data(name=wf, raw_folder=data_folder, columns=DELAY_FEATURES)
        # print(df.describe())
        fn = build_text_data(df=df, folder=data_folder, name=wf)
        df = pd.read_csv(fn)

        # Shuffle the DataFrame
        df = df.sample(frac=1).reset_index(drop=True)

        # split into train/validation/test
        train_ratio, validation_ratio, test_raio = 0.7, 0.1, 0.2
        total_size = len(df)
        train_df = df[: int(total_size * train_ratio)]
        validation_df = df[int(total_size * train_ratio): int(total_size * (train_ratio + validation_ratio))]
        test_df = df[int(total_size * (train_ratio + validation_ratio)):]
        train_df = pd.DataFrame(train_df)
        validation_df = pd.DataFrame(validation_df)
        test_df = pd.DataFrame(test_df)

        # save to local files
        # logging.info(f"save to {data_folder}/{name}")
        train_df.to_csv(f"{data_folder}/{wf}/train.csv", index=False)
        validation_df.to_csv(f"{data_folder}/{wf}/validation.csv", index=False)
        test_df.to_csv(f"{data_folder}/{wf}/test.csv", index=False)
