import json
import os
import os.path as osp


def create_dir(path):
    r""" Create a dir where the processed data will be stored

    Args:
        path (str): Path to create the folder.
    """
    dir_exists = os.path.exists(path)

    if not dir_exists:
        try:
            os.makedirs(path)
            print("The {} directory is created.".format(path))
        except Exception as e:
            print("Error: {}".format(e))
            exit(-1)


def parse_adj(workflow):
    r""" Processing adjacency file.

    Args:
        workflow (str): Workflow name.

    Raises:
        NotImplementedError: No need to process the workflow `all`.

    Returns:
        tuple: (dict, list)
            dict: Dictionary of nodes.
            list: List of directed edges.
    """
    adj_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..", "adjacency_list_dags")
    if workflow == "all":
        raise NotImplementedError
    else:
        adj_file = osp.join(adj_folder, f"{workflow.replace('-', '_')}.json")
    adj = json.load(open(adj_file))

    if workflow in ["predict_future_sales", "pycbc_search"]:
        nodes = {}
        for idx, node_name in enumerate(adj.keys()):
            nodes[node_name] = idx

        edges = []
        for u in adj:
            for v in adj[u]:
                edges.append((nodes[u], nodes[v]))
    else:
        # build dict of node: {node_name: idx}
        nodes = {}
        for idx, node_name in enumerate(adj.keys()):
            if node_name.startswith("create_dir_") or node_name.startswith("cleanup_"):
                node_name = node_name.split("-")[0]
                nodes[node_name] = idx
            else:
                nodes[node_name] = idx

        # build list of edges: [(target, source)]
        edges = []
        for u in adj:
            for v in adj[u]:
                if u.startswith("create_dir_") or u.startswith("cleanup_"):
                    u = u.split("-")[0]
                if v.startswith("create_dir_") or v.startswith("cleanup_"):
                    v = v.split("-")[0]
                edges.append((nodes[u], nodes[v]))

    return nodes, edges
