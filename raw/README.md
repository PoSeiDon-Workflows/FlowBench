# Raw Data

In the raw data you can find:
- *workflows.tar.xz*: the workflows and the configurations used to execute them.
- *archive/workflow-submit-dirs.tar.xz.partX*: the workflow submit directories of all DAG executions, containing workflow management system logs, and provenance data.
- *archive/elasticsearch.tar.xz.partX*: an elasticsearch with the captured workflow events, transfer events and resource utilization traces.

To parse the raw data and generate the parsed dataset you can use the following command.
There are dependencies on Docker and Docker Compose, since elasticsearch runs in a container.
The script untars the tar.xz files, creates a python environment with the needed packages, starts up the elasticsearch instance and goes through the workflow submit directories 
to generated the parsed data. The parsed data will be saved in the ./output folder.

```
bash run_parser.sh
```
