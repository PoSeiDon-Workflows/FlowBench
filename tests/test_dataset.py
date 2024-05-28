import os.path as osp
from flowbench import list_workflows
from flowbench.dataset import FlowDataset, MergeFlowDataset

ROOT = osp.join("/tmp", "data", "poseidon")


def test_workflow():
    for ds in list_workflows():
        dataset = FlowDataset(ROOT, ds, force_reprocess=True, log=True)
        print(dataset)
        assert dataset.name == ds


def test_force_reprocess():
    dataset = FlowDataset(ROOT, "1000genome", force_reprocess=True, log=True)
    print(dataset)
    assert dataset.force_reprocess

    dataset = FlowDataset(ROOT, "1000genome", force_reprocess=False, log=True)
    print(dataset)
    assert not dataset.force_reprocess


def test_binary_labels():
    dataset = FlowDataset(ROOT, "1000genome", binary_labels=True, log=True)
    print(dataset)
    assert dataset.binary_labels

    dataset = FlowDataset(ROOT, "1000genome", binary_labels=False, log=True)
    print(dataset)
    assert not dataset.binary_labels


def test_log():
    dataset = FlowDataset(ROOT, "1000genome", log=True)
    assert dataset.log


def test_node_level():
    dataset = FlowDataset(ROOT, "1000genome", node_level=True)
    assert dataset.node_level
    dataset = FlowDataset(ROOT, "1000genome", node_level=False)
    assert not dataset.node_level


def test_pre_filter():
    # TODO: add test case to filter data by label
    pass


def test_merge_flow_dataset():
    dataset = MergeFlowDataset(ROOT, ["1000genome", "montage"], force_reprocess=True, log=True)
    print(dataset)


# test_1000genome()
# test_montage()
# test_predict_future_sales()
# test_casa_wind_full()
# test_casa_nowcast_full()
# test_variant_calling()

# # test_casa_wind_full()
# dataset = FlowDataset(ROOT, "1000genome", force_reprocess=True, log=False,
#                     pre_filter=lambda data: "cpu" in data.label or "normal" in data.label)
# print(dataset)
# print(dataset.x)
# print(dataset.y)

# print(torch.bincount(dataset.y))
# data, y = dataset.to_graph_level()
# print(data)
# print(y)
# test_merge_flowbench()
# test_1000genome()
