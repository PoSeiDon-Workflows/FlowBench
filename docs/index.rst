.. FlowBench documentation master file, created by
   sphinx-quickstart on Thu May 30 21:26:18 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FlowBench: A Dataset for Computational Workflow Anomaly Detection
=====================================

Flow-Bench is a benchmark dataset for anomaly detection techniques in computational workflows.
Flow-Bench contains workflow execution traces, executed on distributed infrastructure, that include systematically injected anomalies (labeled), and offers both the raw execution logs and a more compact parsed version. 
In this GitHub repository, apart from the logs and traces, you will find sample code to load and process the parsed data using pytorch, as well as, the code used to parse the raw logs and events.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   .. modules
   installation
   dataset
   examples

   flowbench.supervised
   flowbench.unsupervised
   flowbench.nlp

   license
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
