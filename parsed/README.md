# Parsed Data Features

We have parsed the raw logs to create a dataset with the following features for further analysis.

| Field                                          | Type   | Description                                                                                       |
| ---------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------- | 
| node_id                                        | string | Exec. DAG Node ID                                                                                 |
| anomaly_type                                   | string | Anomaly label                                                                                     |
| type                                           | int    | Pegasus job type                                                                                  |
| is_clustered                                   | int    | True if job clustering enabled                                                                    |
| ready                                          | ts     | Epoch ts when all dependencies have been met and job can be dispatched                            |
| pre_script_start                               | ts     | Epoch ts when the pre-script started executing                                                    |
| pre_script_end                                 | ts     | Epoch ts when the pre-script stopped executing                                                    |
| submit                                         | ts     | Epoch ts when the job was submitted to the queue                                                  |
| stage_in_start                                 | ts     | Epoch ts when the data stage in started                                                           |
| stage_in_end                                   | ts     | Epoch ts when the data stage in ended                                                             |
| stage_in_effective_bytes_per_sec               | float  | Bytes written per second (input data)                                                             |
| execute_start                                  | ts     | Epoch ts when the execution starts                                                                |
| execute_end                                    | ts     | Epoch ts when the execution ends                                                                  |
| stage_out_start                                | ts     | Epoch ts when the data stage out started                                                          |
| stage_out_end                                  | ts     | Epoch ts when the data stage out ended                                                            |
| stage_out_effective_bytes_per_sec              | float  | Bytes written per second (output data)                                                            |
| post_script_start                              | ts     | Epoch ts when the post-script started executing                                                   |
| post_script_end                                | ts     | Epoch ts when the post-script ended executing                                                     |
| wms_delay                                      | float  | Composite field estimating the delay introduced by the WMS while preparing the job for submission |
| pre_script_delay                               | float  | Composite field estimating the delay introduced by the pre-script                                 |
| queue_delay                                    | float  | Composite field estimating the time spent in the queue                                            |
| runtime                                        | float  | Total runtime of the job, based on execute start and end                                          |
| post_script_delay                              | float  | Composite field estimating the delay introduced by the post-script                                |
| stage_in_delay                                 | float  | Total time spend staging in data, based on stage in start and end                                 |
| stage_in_bytes                                 | float  | Total bytes staged in (can be casted to int)                                                      |
| stage_out_delay                                | float  | Total time spend staging out data, based on stage out start and end                               |
| stage_out_bytes                                | float  | Total bytes staged out (can be casted to int)                                                     |
| kickstart_user                                 | string | Name of the user submitted the job                                                                |
| kickstart_site                                 | string | Name of the execution site                                                                        |
| kickstart_hostname                             | string | Hostname of the worker node the job executed on                                                   |
| kickstart_transformations                      | string | Mapping of the executable locations                                                               |
| kickstart_executables                          | string | Names of the invoked executables                                                                  |
| kickstart_executables_argv                     | string | Command line arguments used to invoke the executables                                             |
| kickstart_executables_cpu_time                 | float  | Total cpu time                                                                                    |
| kickstart_status                               | int    | The status of the job as marked by Pegasus Kickstart                                              |
| kickstart_executables_exitcode                 | int    | The exitcode of the invoked executable(s)                                                         |
| kickstart_online_iowait                        | float  | Time spent on waiting for io (seconds)                                                            |
| kickstart_online_bytes_read                    | float  | Total bytes read from disk (bytes, can be casted to int)                                          |
| kickstart_online_bytes_written                 | float  | Total bytes written to disk (bytes, can be casted to int)                                         |
| kickstart_online_read_system_calls             | float  | Number of read system calls (can be casted to int)                                                |
| kickstart_online_write_system_calls            | float  | Number of write system calls (can be casted to int)                                               |
| kickstart_online_utime                         | float  | Time spent on user space                                                                          |
| kickstart_online_stime                         | float  | Time spent on kernel space                                                                        |
| kickstart_online_bytes_read_per_second         | float  | Bytes read per second (effective)                                                                 |
| kickstart_online_bytes_written_per_second      | float  | Bytes written per second (effective)
