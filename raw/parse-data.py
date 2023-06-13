#!/usr/bin/env python3

import os
import logging
import pandas as pd
from elasticsearch import Elasticsearch
from time import sleep
import traceback
import json

es = Elasticsearch("http://localhost:9218/")

class WorkflowStatistics:
    dag_file = None
    workflow_name = None
    braindump_file = None
    jobstate_log = None
    dag = None
    root = None
    wf_uuid = None
    root_wf_uuid = None
    wf_started = None
    wf_ended = None
    statistics_df = None
    logger = None
    is_hierarchical = False

    def __init__(self, workflow_name, dag_file, braindump_file, jobstate_log):
        self.logger = logging.getLogger("statistics")
        self.dag_file = dag_file
        self.braindump_file = braindump_file
        self.jobstate_log = jobstate_log
        self.workflow_name = workflow_name

        self.read_dag()
        self.read_braindump()
        self.read_job_log()
    
        return

    def read_dag(self):
        jobs = {}
        allChildNodes = set()
        last_processed_job = None
    
        with open(self.dag_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("JOB"):
                    splitted = line.split()
                    last_processed_job = splitted[1]
                    jobs[splitted[1]] = {
                        "timeline": {"ready": None, "pre_script_start": None, "pre_script_end": None, "submit": None, "execute_start": None, "execute_end": None, "stage_in_start": None, "stage_in_end": None, "stage_out_start": None, "stage_out_end": None, "post_script_start": None, "post_script_end": None}, 
                        "delays": {"wms_delay": None, "pre_script_delay": None, "queue_delay": None, "runtime": None, "post_script_delay": None, "stage_in_delay": None, "stage_out_delay": None}, 
                        "childNodes": [],
                        "parentNodes": [],
                        "is_clustered": 0,
                        "subwf": {"dir": None, "relative_submit_dir": None, "basename": None, "subwf_statistics": None},
                        "stage_in_bytes": None,
                        "stage_in_effective_bytes_per_sec": [],
                        "stage_out_bytes": None,
                        "stage_out_effective_bytes_per_sec": [],
                        "kickstart_data": {}
                    }
                
                if line.startswith("SCRIPT") and last_processed_job.startswith("pegasus-plan"):
                    self.is_hierarchical = True
                    start_index = line.find("--dir") + len("--dir")
                    end_index = line.find("--", start_index+2)
                    jobs[last_processed_job]["subwf"]["dir"] = line[start_index+1:end_index-1]
                    
                    start_index = line.find("--relative-submit-dir") + len("--relative_submit_dir")
                    end_index = line.find("--", start_index+2)
                    jobs[last_processed_job]["subwf"]["relative_submit_dir"] = line[start_index+1:end_index-1]
                    
                    start_index = line.find("--basename") + len("--basename")
                    end_index = line.find("--", start_index+2)
                    jobs[last_processed_job]["subwf"]["basename"] = line[start_index+1:end_index-1]
    
                if line.startswith("PARENT"):
                    splitted = line.split()
                    jobs[splitted[1]]["childNodes"].append(splitted[3])
                    jobs[splitted[3]]["parentNodes"].append(splitted[1])
                    allChildNodes.add(splitted[3])
    
        root = []
        for job in jobs:
            if not job in allChildNodes:
                root.append(job)

        self.root = root
        self.dag = jobs


    def print_adjacency_list(self):
        temp_dict = {}
        
        #if len(self.root) > 1:
        #    temp_dict["dummy_root"] = self.root

        for job in self.dag:
            temp_dict[job] = self.dag[job]["childNodes"]

        print(json.dumps(temp_dict, indent=2))
        

    def read_braindump(self):
        wf_uuid = ""
        root_wf_uuid = ""
        with open(self.braindump_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                splitted = line.split()
                if splitted[0] == "root_wf_uuid:":
                    root_wf_uuid = splitted[1].replace('"','')
                elif splitted[0] == "wf_uuid:":
                    wf_uuid = splitted[1].replace('"', '')
    
        self.root_wf_uuid = root_wf_uuid
        self.wf_uuid = wf_uuid
    
    def read_job_log(self):
        lines = []
    
        wf_started = 0
        wf_ended = 0
    
        with open(self.jobstate_log, 'r') as f:
            lines = f.readlines()
    
        for line in lines:
            splitted = line.split()
            
            if splitted[3] == "DAGMAN_STARTED":
                wf_started = int(splitted[0])
                continue
            elif splitted[3] == "DAGMAN_FINISHED":
                wf_ended = int(splitted[0])
                continue
    
            if splitted[2] == "PRE_SCRIPT_STARTED":
                self.dag[splitted[1]]["timeline"]["pre_script_start"] = int(splitted[0])
            elif splitted[2] == "PRE_SCRIPT_SUCCESS":
                self.dag[splitted[1]]["timeline"]["pre_script_end"] = int(splitted[0])
            elif splitted[2] == "SUBMIT":
                self.dag[splitted[1]]["timeline"]["submit"] = int(splitted[0])
            elif splitted[2] == "EXECUTE":
                self.dag[splitted[1]]["timeline"]["execute_start"] = int(splitted[0])
            elif splitted[2] == "JOB_SUCCESS":
                self.dag[splitted[1]]["timeline"]["execute_end"] = int(splitted[0])
            elif splitted[2] == "POST_SCRIPT_STARTED":
                self.dag[splitted[1]]["timeline"]["post_script_start"] = int(splitted[0])
            elif splitted[2] == "POST_SCRIPT_SUCCESS":
                self.dag[splitted[1]]["timeline"]["post_script_end"] = int(splitted[0])
    
        self.wf_started = wf_started
        self.wf_ended = wf_ended
    
    def calculate_ready(self):
        for job in self.root:
            self.dag[job]["timeline"]["ready"] = self.wf_started
    
        for job in self.dag:
            if job in self.root: continue
    
            parent_finish_ts = []
            for parent in self.dag[job]["parentNodes"]:
                if self.dag[parent]["timeline"]["post_script_end"] == None:
                    parent_finish_ts.append(self.dag[parent]["timeline"]["execute_end"])
                else:
                    parent_finish_ts.append(self.dag[parent]["timeline"]["post_script_end"])
            
            if parent_finish_ts:
                self.dag[job]["timeline"]["ready"] = max(parent_finish_ts)
            else:
                self.dag[job]["timeline"]["ready"] = self.wf_started
    
    def calculate_delays(self):
        for job in self.dag:
            if self.dag[job]["timeline"]["pre_script_start"] is None:
                self.dag[job]["delays"]["wms_delay"] = self.dag[job]["timeline"]["submit"] - self.dag[job]["timeline"]["ready"]
            else:
                self.dag[job]["delays"]["wms_delay"] = self.dag[job]["timeline"]["pre_script_start"] - self.dag[job]["timeline"]["ready"]
            
            if not ((self.dag[job]["timeline"]["pre_script_start"] is None) and (self.dag[job]["timeline"]["pre_script_end"] is None)):
                self.dag[job]["delays"]["pre_script_delay"] = self.dag[job]["timeline"]["pre_script_end"] - self.dag[job]["timeline"]["pre_script_start"]
    
            self.dag[job]["delays"]["queue_delay"] = self.dag[job]["timeline"]["execute_start"] - self.dag[job]["timeline"]["submit"]
            self.dag[job]["delays"]["runtime"] = self.dag[job]["timeline"]["execute_end"] - self.dag[job]["timeline"]["execute_start"]
            if not ((self.dag[job]["timeline"]["stage_in_start"] is None) or (self.dag[job]["timeline"]["stage_in_end"] is None)):
                self.dag[job]["delays"]["stage_in_delay"] = self.dag[job]["timeline"]["stage_in_end"] - self.dag[job]["timeline"]["stage_in_start"]
            if not ((self.dag[job]["timeline"]["stage_out_start"] is None) or (self.dag[job]["timeline"]["stage_out_end"] is None)):
                self.dag[job]["delays"]["stage_out_delay"] = self.dag[job]["timeline"]["stage_out_end"] - self.dag[job]["timeline"]["stage_out_start"]
            if not ((self.dag[job]["timeline"]["post_script_start"] is None) or (self.dag[job]["timeline"]["post_script_end"] is None)):
                self.dag[job]["delays"]["post_script_delay"] = self.dag[job]["timeline"]["post_script_end"] - self.dag[job]["timeline"]["post_script_start"]


    def get_unique_json_records(self, json_records):
        json_records_unique = []
        if not json_records:
            return json_records_unique

        json_records_unique.append(json_records[0])
        for i in range(1, len(json_records)):
            unique = True
            for r in json_records_unique:
                if r == json_records[i]:
                    unique = False
                    break
            if unique:
                json_records_unique.append(json_records[i])

        return json_records_unique


    def retrieve_transfers_from_elastic(self):
        self.logger.info("Retrieving transfer data.")
        es_ids = set()
        
        global es
    
        query = "wf_uuid:\"{0}\" AND event:\"transfer\.inv\.local\"".format(self.wf_uuid)
        
        while True:
            try:
                res = es.search(index="panorama_transfer", q=query, scroll="30m", sort='ts', size=100)
                break
            except:
                self.logger.info("sleeping for 10 seconds and will retry")
                sleep(10)
                pass
    
        num_results = res['hits']['total']
        self.logger.debug(f"Number of results returned by elasticsearch on panorama_transfer: {res['hits']['total']}")
    
        transfer_events = []
        if num_results > 0:
            total_lines = 0
            while total_lines != num_results:
                if res['_scroll_id'] not in es_ids:
                    es_ids.add(res['_scroll_id'])
                for doc in res['hits']['hits']:
                    if doc['_source']['event'] == "transfer.inv.local":
                        transfer_events.append(doc['_source'])
                    else:
                        self.logger.warning("Event is not the expected one !")
                    total_lines += 1
                res = es.scroll(scroll='30m', scroll_id=res['_scroll_id']) 
    
        transfer_events = self.get_unique_json_records(transfer_events)
        transfer_events.sort(key=lambda x: x['transfer_start_time'], reverse=False)
        
        for transfer_event in transfer_events:
            if transfer_event["dag_job_id"].startswith("stage_in"):
                if (self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_start"] is None):
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_start"] = int(transfer_event["transfer_start_time"])
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_end"] = int(transfer_event["transfer_completion_time"])
                    self.dag[transfer_event["dag_job_id"]]["stage_in_bytes"] = int(transfer_event["bytes_transferred"])
                    self.dag[transfer_event["dag_job_id"]]["stage_in_effective_bytes_per_sec"].append((int(transfer_event["effective_bytes_per_second"]),int(transfer_event["transfer_duration"])))
                else:
                    self.logger.info("More than 2 transfer blocks for a stage in job !")
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_end"] = int(transfer_event["transfer_completion_time"])
                    self.dag[transfer_event["dag_job_id"]]["stage_in_bytes"] += int(transfer_event["bytes_transferred"])
                    self.dag[transfer_event["dag_job_id"]]["stage_in_effective_bytes_per_sec"].append((int(transfer_event["effective_bytes_per_second"]),int(transfer_event["transfer_duration"])))
            elif transfer_event["dag_job_id"].startswith("stage_out"):
                if (self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_start"] is None):
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_start"] = int(transfer_event["transfer_start_time"])
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_end"] = int(transfer_event["transfer_completion_time"])
                    self.dag[transfer_event["dag_job_id"]]["stage_out_bytes"] = int(transfer_event["bytes_transferred"])
                    self.dag[transfer_event["dag_job_id"]]["stage_out_effective_bytes_per_sec"].append((int(transfer_event["effective_bytes_per_second"]),int(transfer_event["transfer_duration"])))
                else:
                    self.logger.info("More than 2 transfer blocks for a stage out job !")
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_end"] = int(transfer_event["transfer_completion_time"])
                    self.dag[transfer_event["dag_job_id"]]["stage_out_bytes"] += int(transfer_event["bytes_transferred"])
                    self.dag[transfer_event["dag_job_id"]]["stage_out_effective_bytes_per_sec"].append((int(transfer_event["effective_bytes_per_second"]),int(transfer_event["transfer_duration"])))
            else:
                main_job_start = self.dag[transfer_event["dag_job_id"]]["kickstart_data"]["main_job_start"]
                main_job_end = self.dag[transfer_event["dag_job_id"]]["kickstart_data"]["main_job_end"]

                if (int(transfer_event["transfer_start_time"]) < main_job_start) and (self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_start"] is None):
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_start"] = int(transfer_event["transfer_start_time"])
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_end"] = int(transfer_event["transfer_completion_time"])
                    self.dag[transfer_event["dag_job_id"]]["stage_in_bytes"] = int(transfer_event["bytes_transferred"])
                    self.dag[transfer_event["dag_job_id"]]["stage_in_effective_bytes_per_sec"].append((int(transfer_event["effective_bytes_per_second"]),int(transfer_event["transfer_duration"])))
                elif int(transfer_event["transfer_start_time"]) < main_job_start:
                    self.dag[transfer_event["dag_job_id"]]["stage_in_bytes"] += int(transfer_event["bytes_transferred"])
                    self.dag[transfer_event["dag_job_id"]]["stage_in_effective_bytes_per_sec"].append((int(transfer_event["effective_bytes_per_second"]),int(transfer_event["transfer_duration"])))
                    if self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_end"] < int(transfer_event["transfer_completion_time"]):
                        self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_in_end"] = int(transfer_event["transfer_completion_time"])
                elif (int(transfer_event["transfer_start_time"]) >= main_job_end) and (self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_start"] is None):
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_start"] = int(transfer_event["transfer_start_time"])
                    self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_end"] = int(transfer_event["transfer_completion_time"])
                    self.dag[transfer_event["dag_job_id"]]["stage_out_bytes"] = int(transfer_event["bytes_transferred"])
                    self.dag[transfer_event["dag_job_id"]]["stage_out_effective_bytes_per_sec"].append((int(transfer_event["effective_bytes_per_second"]),int(transfer_event["transfer_duration"])))
                elif int(transfer_event["transfer_start_time"]) >= main_job_end:
                    self.dag[transfer_event["dag_job_id"]]["stage_out_bytes"] += int(transfer_event["bytes_transferred"])
                    self.dag[transfer_event["dag_job_id"]]["stage_out_effective_bytes_per_sec"].append((int(transfer_event["effective_bytes_per_second"]),int(transfer_event["transfer_duration"])))
                    if self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_end"] < int(transfer_event["transfer_completion_time"]):
                        self.dag[transfer_event["dag_job_id"]]["timeline"]["stage_out_end"] = int(transfer_event["transfer_completion_time"])
                else:
                    self.logger.warning(f"Out of order transfer events for job {transfer_event['dag_job_id']}!")
                    self.logger.debug(f"main_job_start: {main_job_start}")
                    self.logger.debug(f"main_job_end: {main_job_end}")
                    self.logger.debug(f"transfer_event:\n{json.dumps(transfer_event, indent=2)}")
                    raise Exception(f"Out of order transfer event, probably there was a retry!")


    def retrieve_kickstart_from_elasticsearch(self):
        self.logger.info("Retrieving kickstart data.")
        es_ids = set()

        global es
    
        query = "xwf_id:\"{0}\" AND event:\"stampede\.job_inst\.composite\"".format(self.wf_uuid)
        
        while True:
            try:
                res = es.search(index="panorama_stampede", q=query, scroll="30m", sort='ts', size=100)
                break
            except:
                self.logger.info("sleeping for 10 seconds and will retry")
                sleep(10)
                pass
        
        num_results = res['hits']['total']
        self.logger.debug(f"Number of results returned by elasticsearch on panorama_stampede(stampede.job_inst.composite) index: {res['hits']['total']}")
        
        composite_events = []
        if num_results > 0:
            total_lines = 0
            while total_lines != num_results:
                if res['_scroll_id'] not in es_ids:
                    es_ids.add(res['_scroll_id'])
                for doc in res['hits']['hits']:
                    if doc['_source']['event'] == "stampede.job_inst.composite":
                        composite_events.append(doc['_source'])
                    else:
                        self.logger.warning("Event is not the expected one !")
                    total_lines += 1
                res = es.scroll(scroll='30m', scroll_id=res['_scroll_id'])
        else:
            return
    
        composite_events = self.get_unique_json_records(composite_events)
        composite_events.sort(key=lambda x: x['wf_ts'], reverse=False)
        
        query = "xwf_id:\"{0}\" AND event:\"stampede\.inv\.start\"".format(self.wf_uuid)
    
        while True:
            try:
                res = es.search(index="panorama_stampede", q=query, scroll="30m", sort='ts', size=100)
                break
            except:
                sleep(10)
                pass
    
        num_results = res['hits']['total']
        self.logger.debug(f"Number of results returned by elasticsearch on panorama_stampede(stampede.inv.start) index: {res['hits']['total']}")
        
        invocation_start_events = []
        if num_results > 0:
            total_lines = 0
            while total_lines != num_results:
                if res['_scroll_id'] not in es_ids:
                    es_ids.add(res['_scroll_id'])
                for doc in res['hits']['hits']:
                    if doc['_source']['event'] == "stampede.inv.start":
                        if doc['_source']['inv_id'] > -1:
                            invocation_start_events.append(doc['_source'])
                    else:
                        self.logger.warning("Event is not the expected one !")
                    total_lines += 1
                res = es.scroll(scroll='30m', scroll_id=res['_scroll_id'])
        else:
            return
    
        invocation_start_events = self.get_unique_json_records(invocation_start_events)
        invocation_start_events.sort(key=lambda x: x['wf_ts'], reverse=False)
        
        query = "xwf_id:\"{0}\" AND event:\"stampede\.inv\.end\"".format(self.wf_uuid)
    
        while True:
            try:
                res = es.search(index="panorama_stampede", q=query, scroll="30m", sort='ts', size=100)
                break
            except:
                sleep(10)
                pass
    
        num_results = res['hits']['total']
        self.logger.debug(f"Number of results returned by elasticsearch on panorama_stampede(stampede.inv.end) index: {res['hits']['total']}")
        
        invocation_end_events = []
        if num_results > 0:
            total_lines = 0
            while total_lines != num_results:
                if res['_scroll_id'] not in es_ids:
                    es_ids.add(res['_scroll_id'])
                for doc in res['hits']['hits']:
                    if doc['_source']['event'] == "stampede.inv.end":
                        if doc['_source']['inv_id'] > -1:
                            invocation_end_events.append(doc['_source'])
                    else:
                        self.logger.warning("Event is not the expected one !")
                    total_lines += 1
                res = es.scroll(scroll='30m', scroll_id=res['_scroll_id'])
        else:
            return
    
        invocation_end_events = self.get_unique_json_records(invocation_end_events)
        invocation_end_events.sort(key=lambda x: x['wf_ts'], reverse=False)
    

        query = "wf_uuid:\"{0}\" AND event:\"kickstart\.inv\.online\"".format(self.wf_uuid)
    
        while True:
            try:
                res = es.search(index="panorama_kickstart", q=query, scroll="30m", sort='ts', size=100)
                break
            except:
                sleep(10)
                pass
    
        num_results = res['hits']['total']
        self.logger.debug(f"Number of results returned by elasticsearch on panorama_kickstart(kickstart.inv.online) index: {res['hits']['total']}")
        
        kickstart_online_events = []
        if num_results > 0:
            total_lines = 0
            while total_lines != num_results:
                if res['_scroll_id'] not in es_ids:
                    es_ids.add(res['_scroll_id'])
                for doc in res['hits']['hits']:
                    if doc['_source']['event'] == "kickstart.inv.online":
                        kickstart_online_events.append(doc['_source'])
                    else:
                        self.logger.warning("Event is not the expected one !")
                    total_lines += 1
                res = es.scroll(scroll='30m', scroll_id=res['_scroll_id'])
    
        kickstart_online_events = self.get_unique_json_records(kickstart_online_events)
        kickstart_online_events.sort(key=lambda x: x['ts'], reverse=False)
        
        for job in self.dag:
            j_comp = [e for e in composite_events if e["job_id"] == job]
            j_inv_start = [e for e in invocation_start_events if e["job_id"] == job]
            j_inv_end = [e for e in invocation_end_events if e["job_id"] == job]
            j_inv_end.sort(key=lambda x: x['inv_id'], reverse=False)
            j_online_events = [e for e in kickstart_online_events if e["dag_job_id"] == job]
    
            if not "hostname" in j_comp[0]:
                self.logger.warning(f"Job ({job}) composite event doesn't contain a hostname")
                j_comp[0]["hostname"] = "poseidon-submit"

            self.dag[job]["kickstart_data"]["user"] = j_comp[0]["user"]
            self.dag[job]["kickstart_data"]["site"] = j_comp[0]["site"]
            self.dag[job]["kickstart_data"]["hostname"] = j_comp[0]["hostname"]
            self.dag[job]["kickstart_data"]["status"] = j_comp[0]["status"]
            
            self.dag[job]["kickstart_data"]["main_job_start"] = min([int(e["ts"]) for e in j_inv_start])
            self.dag[job]["kickstart_data"]["main_job_end"] = max([int(e["ts"]) for e in j_inv_end])
    
            self.dag[job]["kickstart_data"]["transformations"] = []
            self.dag[job]["kickstart_data"]["executables"] = []
            self.dag[job]["kickstart_data"]["executables_argv"] = []
            self.dag[job]["kickstart_data"]["cpu_time"] = []
            self.dag[job]["kickstart_data"]["exitcode"] = []
            if len(j_inv_end) > 1:
                self.dag[job]["is_clustered"] = 1
            else:
                #not supported for clustered jobs yet
                if len(j_online_events) > 0: 
                    self.dag[job]["kickstart_data"]["iowait"] = j_online_events[-1]["iowait"]
                    self.dag[job]["kickstart_data"]["bytes_read"] = j_online_events[-1]["rchar"]
                    self.dag[job]["kickstart_data"]["bytes_written"] = j_online_events[-1]["wchar"]
                    self.dag[job]["kickstart_data"]["read_system_calls"] = j_online_events[-1]["syscr"]
                    self.dag[job]["kickstart_data"]["write_system_calls"] = j_online_events[-1]["syscw"]
                    self.dag[job]["kickstart_data"]["utime"] = j_online_events[-1]["utime"]
                    self.dag[job]["kickstart_data"]["stime"] = j_online_events[-1]["stime"]

                    #calculate bytes_read_per_second
                    #calculate bytes_written_per_second

                    current_bytes_read = 0
                    current_bytes_written = 0
                    change_in_reads = 0
                    change_in_writes = 0
                    bytes_read_per_second = 0
                    bytes_written_per_second = 0
                    online_monitoring_interval = 10.0 #fixed for now
                    for e in j_online_events:
                        if current_bytes_read < e["rchar"]:
                            d_read = e["rchar"] - current_bytes_read
                            change_in_reads += 1
                            bytes_read_per_second += d_read*1.0/online_monitoring_interval
                            current_bytes_read = e["rchar"]
                        if current_bytes_written < e["wchar"]:
                            d_written = e["wchar"] - current_bytes_written
                            change_in_writes += 1
                            bytes_written_per_second += d_written*1.0/online_monitoring_interval
                            current_bytes_written = e["wchar"]

                    self.dag[job]["kickstart_data"]["avg_bytes_read_per_second"] = bytes_read_per_second/change_in_reads if change_in_reads > 0 else None
                    self.dag[job]["kickstart_data"]["avg_bytes_written_per_second"] = bytes_written_per_second/change_in_writes if change_in_writes > 0 else None

            for e in j_inv_end:
                self.dag[job]["kickstart_data"]["transformations"].append(e["transformation"])
                self.dag[job]["kickstart_data"]["executables"].append(e["executable"])
                if "argv" in e:
                    self.dag[job]["kickstart_data"]["executables_argv"].append(e["argv"].strip())
                if "remote_cpu_time" in e:
                    self.dag[job]["kickstart_data"]["cpu_time"].append(e["remote_cpu_time"])
                else:
                    self.dag[job]["kickstart_data"]["cpu_time"].append(0)
                    self.logger.warning(f"Job ({job}) invocation end event doesn't contain remote_cpu_time")
                self.dag[job]["kickstart_data"]["exitcode"].append(e["exitcode"])
            
    
    def populate_extended_stats(self):
        self.retrieve_kickstart_from_elasticsearch()
        if not self.dag[self.root[0]]["kickstart_data"]:
            raise Exception(f"No kickstart data for {self.workflow_name} with uuid {self.wf_uuid}.")
        self.retrieve_transfers_from_elastic()
        self.calculate_ready()
        self.calculate_delays()


    def populate_subworkflow_statistics(self):
        for job in self.dag:
            if job.startswith("pegasus-plan"):
                submit_dir = self.dag_file[:self.dag_file.find("/submit/")+7]
                run_dir = os.path.join(submit_dir, self.dag[job]["subwf"]["relative_submit_dir"])
                dag_file = os.path.join(run_dir, f"{self.dag[job]['subwf']['basename']}.dag")
                braindump_file = os.path.join(run_dir, "braindump.yml")
                jobstate_log = os.path.join(run_dir, "jobstate.log")

                self.dag[job]["subwf"]["subwf_statistics"] = WorkflowStatistics(self.dag[job]["subwf"]["basename"], dag_file, braindump_file, jobstate_log)
                self.dag[job]["subwf"]["subwf_statistics"].populate_extended_stats()
                

    def flatten_hierarchical_statistics(self):
        temp_dict = {}
        for job in self.dag:
            if job.startswith("pegasus-plan"):
                h_dag = self.dag[job]["subwf"]["subwf_statistics"].dag
                for h_job in h_dag:
                    temp_dict[f"{h_job}_{job}"] = h_dag[h_job]
                    if not temp_dict[f"{h_job}_{job}"]["parentNodes"]:
                        temp_dict[f"{h_job}_{job}"]["parentNodes"] = self.dag[job]["parentNodes"]
                        for parent in temp_dict[f"{h_job}_{job}"]["parentNodes"]:
                            self.dag[parent]["childNodes"].append(f"{h_job}_{job}")
                    else:
                        temp_list = []
                        for parent in temp_dict[f"{h_job}_{job}"]["parentNodes"]:
                            temp_list.append(f"{parent}_{job}")
                        temp_dict[f"{h_job}_{job}"]["parentNodes"] = temp_list
                    
                    if not temp_dict[f"{h_job}_{job}"]["childNodes"]: 
                        temp_dict[f"{h_job}_{job}"]["childNodes"] = self.dag[job]["childNodes"]
                        for child in temp_dict[f"{h_job}_{job}"]["childNodes"]:
                            self.dag[child]["parentNodes"].append(f"{h_job}_{job}") 
                    else:
                        temp_list = []
                        for child in temp_dict[f"{h_job}_{job}"]["childNodes"]:
                            temp_list.append(f"{child}_{job}")
                        temp_dict[f"{h_job}_{job}"]["childNodes"] = temp_list

        self.dag = {**self.dag, **temp_dict}


    def prepare_pandas_json(self, anomaly_type, anomaly_nodes):
        pandas_json = {}
        for job in self.dag:
            self.logger.debug(f"Processing job: {job}")
            temp_dict = {}
            temp_dict.update(self.dag[job]["timeline"])
            temp_dict.update(self.dag[job]["delays"])
            
            if job.startswith(("stage_in", "stage_out", "stage_inter")):
                temp_dict["type"] = "transfer"
            elif job.startswith(("create", "cleanup", "register")):
                temp_dict["type"] = "auxiliary"
            elif job.startswith("pegasus-plan"):
                temp_dict["type"] = "subwf"
            else:
                temp_dict["type"] = "compute"
            
            temp_dict["is_clustered"] = self.dag[job]["is_clustered"]
            
            temp_dict["stage_in_bytes"] = self.dag[job]["stage_in_bytes"]
            total_time_weight = sum([t for (b, t) in self.dag[job]["stage_in_effective_bytes_per_sec"]])*1.0
            temp_dict["stage_in_effective_bytes_per_sec"] = sum([(b*t*1.0)/total_time_weight for (b, t) in self.dag[job]["stage_in_effective_bytes_per_sec"]])
            
            temp_dict["stage_out_bytes"] = self.dag[job]["stage_out_bytes"]
            total_time_weight = sum([t for (b, t) in self.dag[job]["stage_out_effective_bytes_per_sec"]])*1.0
            temp_dict["stage_out_effective_bytes_per_sec"] = sum([(b*t*1.0)/total_time_weight for (b, t) in self.dag[job]["stage_out_effective_bytes_per_sec"]])
            
            temp_dict["kickstart_user"] = self.dag[job]["kickstart_data"]["user"] 
            temp_dict["kickstart_site"] = self.dag[job]["kickstart_data"]["site"]
            temp_dict["kickstart_hostname"] = self.dag[job]["kickstart_data"]["hostname"]
            temp_dict["kickstart_status"] = self.dag[job]["kickstart_data"]["status"]
            
            try:
                temp_dict["kickstart_online_iowait"] = self.dag[job]["kickstart_data"]["iowait"]
                temp_dict["kickstart_online_bytes_read"] = self.dag[job]["kickstart_data"]["bytes_read"]
                temp_dict["kickstart_online_bytes_written"] = self.dag[job]["kickstart_data"]["bytes_written"]
                temp_dict["kickstart_online_read_system_calls"] = self.dag[job]["kickstart_data"]["read_system_calls"]
                temp_dict["kickstart_online_write_system_calls"] = self.dag[job]["kickstart_data"]["write_system_calls"]
                temp_dict["kickstart_online_utime"] = self.dag[job]["kickstart_data"]["utime"]
                temp_dict["kickstart_online_stime"] = self.dag[job]["kickstart_data"]["stime"]
                temp_dict["kickstart_online_bytes_read_per_second"] = self.dag[job]["kickstart_data"]["avg_bytes_read_per_second"]
                temp_dict["kickstart_online_bytes_written_per_second"] = self.dag[job]["kickstart_data"]["avg_bytes_written_per_second"]
            except Exception as e:
                temp_dict["kickstart_online_iowait"] = None
                temp_dict["kickstart_online_bytes_read"] = None
                temp_dict["kickstart_online_bytes_written"] = None 
                temp_dict["kickstart_online_read_system_calls"] = None 
                temp_dict["kickstart_online_write_system_calls"] = None 
                temp_dict["kickstart_online_utime"] = None
                temp_dict["kickstart_online_stime"] = None
                temp_dict["kickstart_online_bytes_read_per_second"] = None
                temp_dict["kickstart_online_bytes_written_per_second"] = None
    
            temp_dict["kickstart_transformations"] = ";".join(self.dag[job]["kickstart_data"]["transformations"])
            temp_dict["kickstart_executables"] = ";".join(self.dag[job]["kickstart_data"]["executables"])
            temp_dict["kickstart_executables_argv"] = ";".join(self.dag[job]["kickstart_data"]["executables_argv"])
            temp_dict["kickstart_executables_cpu_time"] = sum(self.dag[job]["kickstart_data"]["cpu_time"])
            
            temp_dict["kickstart_executables_exitcode"] = max(self.dag[job]["kickstart_data"]["exitcode"])

            if temp_dict["kickstart_hostname"] in anomaly_nodes:
                temp_dict["anomaly_type"] = anomaly_type
            else:
                temp_dict["anomaly_type"] = "None"
            pandas_json[job] = temp_dict
    
        return pandas_json

    def prepare_pandas_df(self, anomaly_type, anomaly_nodes):
        pandas_json = self.prepare_pandas_json(anomaly_type, anomaly_nodes)
        df = pd.read_json(json.dumps(pandas_json, indent=2), orient="index")
        dtypes = {"is_clustered": "int64",
                "ready": "float64", 
                "pre_script_start": "float64", 
                "pre_script_end": "float64", 
                "submit": "float64", 
                "execute_start": "float64",
                "execute_end": "float64",
                "stage_in_start": "float64",
                "stage_in_end": "float64", 
                "stage_in_effective_bytes_per_sec": "float64",
                "stage_out_start": "float64",
                "stage_out_end": "float64",
                "stage_out_effective_bytes_per_sec": "float64",
                "post_script_start": "float64", 
                "post_script_end": "float64", 
                "wms_delay": "float64", 
                "pre_script_delay": "float64", 
                "queue_delay": "float64", 
                "runtime": "float64",
                "stage_in_delay": "float64",
                "stage_out_delay": "float64",
                "post_script_delay": "float64", 
                "kickstart_executables_cpu_time": "float64",
                "kickstart_executables_exitcode": "int64",
                "type": "object",
                "anomaly_type": "object"
        }
        
        df = df.astype(dtypes)
        df = df.sort_values(by="ready")
        df = df[[
            "anomaly_type",
            "type",
            "is_clustered",
            "ready",
            "pre_script_start",
            "pre_script_end",
            "submit",
            "stage_in_start",
            "stage_in_end",
            "stage_in_effective_bytes_per_sec",
            "execute_start",
            "execute_end",
            "stage_out_start",
            "stage_out_end",
            "stage_out_effective_bytes_per_sec",
            "post_script_start",
            "post_script_end",
            "wms_delay",
            "pre_script_delay",
            "queue_delay",
            "runtime",
            "post_script_delay",
            "stage_in_delay",
            "stage_in_bytes",
            "stage_out_delay",
            "stage_out_bytes",
            "kickstart_user",
            "kickstart_site", 
            "kickstart_hostname",
            "kickstart_transformations",
            "kickstart_executables", 
            "kickstart_executables_argv",
            "kickstart_executables_cpu_time",
            "kickstart_status",
            "kickstart_executables_exitcode",
            "kickstart_online_iowait",
            "kickstart_online_bytes_read",
            "kickstart_online_bytes_written",
            "kickstart_online_read_system_calls",
            "kickstart_online_write_system_calls",
            "kickstart_online_utime",
            "kickstart_online_stime",
            "kickstart_online_bytes_read_per_second",
            "kickstart_online_bytes_written_per_second"
        ]]

        self.statistics_df = df

    def write_to_csv(self, output_file):
        self.statistics_df.to_csv(output_file, float_format='%.1f')


class Statistics:
    logger = None
    experiments_log = None
    workflows_config = None
    output_dir = "output"

    def __init__(self, experiments_log, workflow_list):
        self.logger = logging.getLogger("statistics")
        self.experiments_log = experiments_log
        self.workflow_list = workflow_list
        return
    
    def find_experiment_type(self, workflow_name, wf_started, wf_ended):

        log_line = self.experiments_log[(self.experiments_log["workflow"] == workflow_name) & (self.experiments_log["start_time"] <= wf_started) & (self.experiments_log["end_time"] >= wf_ended)]

        if log_line.empty:
            raise Exception("Workflow couldn't be matched with the log entries.")
        if len(log_line) > 1:
            self.logger.warning("Workflow matched multiple log entries. The last one will be used.")

        log_line_string = log_line.to_string(header=False, index=False, index_names=False)
        self.logger.debug(f"Found line in log: {log_line_string}")
        
        experiment_type = log_line.iloc[-1]["experiment_type"]
        anomaly_nodes = log_line.iloc[-1]["workers_with_anomaly"]

        if type(anomaly_nodes) is float and pd.isna(anomaly_nodes) and experiment_type != "normal":
            raise Exception("Experiment type is {experiment_type} but there are no anomaly nodes.")
        elif experiment_type == "normal":
            anomaly_nodes = ""
        else:
            anomaly_nodes = anomaly_nodes.split()
        return (experiment_type, anomaly_nodes)
    
    def generate(self):
        experiment_types = list(self.experiments_log["experiment_type"].unique())
    
        for experiment_type in experiment_types:
            os.makedirs(os.path.join(self.output_dir, experiment_type), exist_ok = True)

        for workflow_name in self.workflow_list:
            workflow_submit_dir = self.workflow_list[workflow_name]["submit_dir"]
            workflow_submit_dir = os.path.join(workflow_submit_dir, workflow_name)

            if not os.path.exists(workflow_submit_dir):
                self.logger.warning(f"Workflow submit dir doesn't exist ({workflow_submit_dir})")
                continue

            for sub_folder in os.listdir(workflow_submit_dir):
                run_dir = os.path.join(workflow_submit_dir, sub_folder)
                dag_file = os.path.join(run_dir, f"{workflow_name}-0.dag")
                braindump_file = os.path.join(run_dir, "braindump.yml")
                jobstate_log = os.path.join(run_dir, "jobstate.log")

                workflow_statistics = WorkflowStatistics(workflow_name, dag_file, braindump_file, jobstate_log)
                
                try:
                    (experiment_type, anomaly_nodes) = self.find_experiment_type(workflow_name, workflow_statistics.wf_started, workflow_statistics.wf_ended)
                except Exception as e:
                    self.logger.warning(f"{workflow_name} ({run_dir}): {e}")
                    self.logger.debug(traceback.format_exc())
                    continue

                output_file = os.path.join(self.output_dir, experiment_type, f"{workflow_name}-{sub_folder}.csv")
                self.logger.debug(f"Output file is: {output_file}.")

                if os.path.exists(output_file):
                    self.logger.info(f"Output file exists: {output_file}.")
                    self.logger.info("Skipping...")
                    continue
                
                try:
                    workflow_statistics.populate_extended_stats()
                except Exception as e:
                    self.logger.warning(f"{workflow_name} ({run_dir}): {e}")
                    self.logger.debug(traceback.format_exc())
                    self.logger.info("Skipping...")
                    continue

                if workflow_statistics.is_hierarchical:
                    try:
                        workflow_statistics.populate_subworkflow_statistics()
                        workflow_statistics.flatten_hierarchical_statistics()
                    except Exception as e:
                        self.logger.warning(e)
                        self.logger.info("Skipping...")
                        continue
                
                workflow_statistics.prepare_pandas_df(experiment_type, anomaly_nodes)
                workflow_statistics.write_to_csv(output_file)
                sleep(2)
        return


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("statistics")
    logger.setLevel(logging.INFO)

    experiments_log = pd.read_csv("experiments.csv")
    workflow_list = {
        "1000-genome": {"submit_dir": "workflow-submit-dirs/1000genome-workflow/submit/poseidon/pegasus"},
        "montage": {"submit_dir": "workflow-submit-dirs/montage-workflow-v3/submit/poseidon/pegasus"},
        "predict-future-sales": {"submit_dir": "workflow-submit-dirs/predict-future-sales-workflow/submit/poseidon/pegasus"}
    }

    stats = Statistics(experiments_log, workflow_list)
    stats.generate()
