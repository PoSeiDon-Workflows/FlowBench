| Field                                          | Description                                                                                       |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------- | 
| node_id                                        | Exec. DAG Node ID                                                                                 |
| anomaly_type                                   | Anomaly label                                                                                     |
| type                                           | Pegasus job type                                                                                  |
| is_clustered                                   | True if job clustering enabled                                                                    |
| ready                                          | Epoch ts when all dependencies have been met and job can be dispatched                            |
| pre_script_start                               | Epoch ts when the pre-script started executing                                                    |
| pre_script_end                                 | Epoch ts when the pre-script stopped executing                                                    |
| submit                                         | Epoch ts when the job was submitted to the queue                                                  |
| stage_in_start                                 | Epoch ts when the data stage in started                                                           |
| stage_in_end                                   | Epoch ts when the data stage in ended                                                             |
| stage_in_effective_bytes_per_sec               | Bytes written per second (input data)                                                             |
| execute_start                                  | Epoch ts when the execution starts                                                                |
| execute_end                                    | Epoch ts when the execution ends                                                                  |
| stage_out_start                                | Epoch ts when the data stage out started                                                          |
| stage_out_end                                  | Epoch ts when the data stage out ended                                                            |
| stage_out_effective_bytes_per_sec              | Bytes written per second (output data)                                                            |
| post_script_start                              | Epoch ts when the post-script started executing                                                   |
| post_script_end                                | Epoch ts when the post-script ended executing                                                     |
| wms_delay                                      | Composite field estimating the delay introduced by the WMS while preparing the job for submission |
| pre_script_delay                               | Composite field estimating the delay introduced by the pre-script                                 |
| queue_delay                                    | Composite field estimating the time spent in the queue                                            |
| runtime                                        | Total runtime of the job, based on execute start and end                                          |
| post_script_delay                              | Composite field estimating the delay introduced by the post-script                                |
| stage_in_delay                                 | Total time spend staging in data, based on stage in start and end                                 |
| stage_in_bytes                                 | Total bytes staged in                                                                             |
| stage_out_delay                                | Total time spend staging out data, based on stage out start and end                               |
| stage_out_bytes                                | Total bytes staged out                                                                            |
| kickstart_user                                 | Name of the user submitted the job                                                                |
| kickstart_site                                 | Name of the execution site                                                                        |
| kickstart_hostname                             | Hostname of the worker node the job executed on                                                   |
| kickstart_transformations                      | Mapping of the executable locations                                                               |
| kickstart_executables                          | Names of the invoked executables                                                                  |
| kickstart_executables_argv                     | Command line arguments used to invoke the executables                                             |
| kickstart_executables_cpu_time                 | Total cpu time                                                                                    |
| kickstart_status                               | The status of the job as marked by Pegasus Kickstart                                              |
| kickstart_executables_exitcode                 | The exitcode of the invoked executable(s)                                                         |
| kickstart_online_iowait                        | Time spent on waiting for io (seconds)                                                            |
| kickstart_online_bytes_read                    | Total bytes read from disk (bytes)                                                                |
| kickstart_online_bytes_written                 | Total bytes written to disk (bytes)                                                               |
| kickstart_online_read_system_calls             | Number of read system calls                                                                       |
| kickstart_online_write_system_calls            | Number of write system calls                                                                      |
| kickstart_online_utime                         | Time spent on user space                                                                          |
| kickstart_online_stime                         | Time spent on kernel space                                                                        |
| kickstart_online_bytes_read_per_second         | Bytes read per second (effective)                                                                 |
| kickstart_online_bytes_written_per_second      | Bytes written per second (effective)
