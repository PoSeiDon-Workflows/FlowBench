from setuptools import find_packages, setup

setup(
    name="flowbench",
    version="0.2.0",
    author="PoSeiDon Team",
    summary="""Flow-Bench is a benchmark dataset for anomaly detection techniques in computational workflows.
      Flow-Bench contains workflow execution traces, executed on distributed infrastructure, that include systematically
      injected anomalies (labeled), and offers both the raw execution logs and a more compact parsed version. In this
      GitHub repository, apart from the logs and traces, you will find sample code to load and process the parsed data
      using pytorch, as well as, the code used to parse the raw logs and events.""",
    license="MIT",
    author_email="anonymous",
    packages=find_packages(exclude=["tests", "raw", "results", "log", "flowbench.egg-info"]),
    extras_require={
        'pyg': ['pytorch_geometric'],
        'pygod': ['pygod'],
        'pyod': ['pyod'],
    }
)
