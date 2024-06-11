""" FlowBench: A benchmarking framework for scientific workflows.

Author: PoSeiDon Team
License: MIT
"""


def list_workflows():
    r""" List of available workflows in FlowBench.

    Returns:
        list: List of workflow names.
    """
    return [
        "1000genome",
        "casa_nowcast_full",
        "casa_wind_full",
        "eht_difmap",
        "eht_imaging",
        "eht_smili",
        "montage",
        "predict_future_sales",
        "pycbc_inference",
        "pycbc_search",
        "somospie",
        "variant_calling",
    ]
