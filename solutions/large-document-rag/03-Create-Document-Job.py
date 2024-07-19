# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook creates a job to load, process, and sync .txt files from the Volume to the online table and vector index.
# MAGIC
# MAGIC Steps to Execute:
# MAGIC 1. Ensure your parameters are set according to your setup.
# MAGIC 2. Note that this job will be run by the Service Principal. You must add the git credentials to that Service Principal.
# MAGIC
# MAGIC Key Points:
# MAGIC - The main code to be executed is located in 03b-Load-Documents.
# MAGIC - The job is triggered when new files are dropped into the source volume path.
# MAGIC - The task will launch a cluster to chunk the documents and load them into parent and child tables.
# MAGIC - Two DLT (Data Lakehouse Tool) pipelines will follow:
# MAGIC   1. Update the child vector search index.
# MAGIC   2. Update the parent online table.
# MAGIC
# MAGIC Please follow these instructions carefully to ensure proper execution.
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Please change parameters according to your setup!
git_provider = "gitHub"
git_repo = "https://github.com/AndreasJaeck/databricks-field-genai.git"
git_branch = "feature/trigger-run-with-sp-from-git-source"
notebook_path = "solutions/large-document-rag/03b-Load-Documents"

# Be aware that this job will be run by the Service Principal and that you need to add the git credentials to that Service Principal 

# COMMAND ----------

import os
import mlflow

from databricks.sdk import WorkspaceClient

#Get the conf from the local conf file
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

databricks_resources = model_config.get("databricks_resources")
secrets_config = model_config.get("secrets_config")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

# Get Service Principal Token from secrets service 
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(secrets_config.get("secret_scope"), secrets_config.get("secret_key"))
# Get the host from the configuration
os.environ["DATABRICKS_HOST"] = databricks_resources.get("host")

w = WorkspaceClient(host=databricks_resources.get("host"), token=os.environ['DATABRICKS_TOKEN'])
sp_id = w.current_user.me().emails[0].value
print(f"Service Principal ID: {sp_id}")

# COMMAND ----------

from databricks.sdk.service import jobs

# Be aware that this job will be run by the Service Principal and that you need to add the git credentials to that Service Principal 

# This job config will create a job that is triggered when new files are droped into the source volume path. Task will launch a cluster that is chunking the documents and loading them into parent and child tables. Followed by two dlt pieplines to update the child vector search index and the parent online table. 

# Please update run_as user_name according to your setup.
job_config = {
  "name": f"{databricks_resources.get('schema')}_update_documents",
  "email_notifications": {
    "no_alert_for_skipped_runs": False
  },
  "webhook_notifications": {},
  "timeout_seconds": 0,
  "trigger": {
    "pause_status": "UNPAUSED",
    "file_arrival": {
      "url": f"{databricks_resources.get('source_volume_path')}/"
    }
  },
  "max_concurrent_runs": 1,
  "tasks": [
    {
      "task_key": f"{databricks_resources.get('schema')}_update_tables",
      "run_if": "ALL_SUCCESS",
      "notebook_task": {
        "notebook_path": notebook_path,
        "source": "GIT"
      },
      "job_cluster_key": f"{databricks_resources.get('schema')}_update_cluster",
      "timeout_seconds": 0,
      "email_notifications": {},
      "notification_settings": {
        "no_alert_for_skipped_runs": False,
        "no_alert_for_canceled_runs": False,
        "alert_on_last_attempt": False
      },
      "webhook_notifications": {}
    },
    {
      "task_key": f"Update_{databricks_resources.get('schema')}_child_splits_index",
      "depends_on": [
        {
          "task_key": f"{databricks_resources.get('schema')}_update_tables",
        }
      ],
      "run_if": "ALL_SUCCESS",
      "pipeline_task": {
        "pipeline_id": w.vector_search_indexes.get_index(databricks_resources.get("vector_search_index")).delta_sync_index_spec.pipeline_id,
        "full_refresh": False
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "notification_settings": {
        "no_alert_for_skipped_runs": False,
        "no_alert_for_canceled_runs": False,
        "alert_on_last_attempt": False
      },
      "webhook_notifications": {}
    },
    {
      "task_key": f"Update_{databricks_resources.get('schema')}_parent_splits_online",
      "depends_on": [
        {
          "task_key": f"{databricks_resources.get('schema')}_update_tables"
        }
      ],
      "run_if": "ALL_SUCCESS",
      "pipeline_task": {
        "pipeline_id": w.online_tables.get(f"{databricks_resources.get('parent_table')}_online").spec.pipeline_id,
        "full_refresh": False
      },
      "timeout_seconds": 0,
      "email_notifications": {},
      "notification_settings": {
        "no_alert_for_skipped_runs": False,
        "no_alert_for_canceled_runs": False,
        "alert_on_last_attempt": False
      },
      "webhook_notifications": {}
    }
  ],
  "job_clusters": [
    {
      "job_cluster_key": f"{databricks_resources.get('schema')}_update_cluster",
      "new_cluster": {
        "spark_version": "15.3.x-cpu-ml-scala2.12",
        "aws_attributes": {
          "first_on_demand": 1,
          "availability": "SPOT_WITH_FALLBACK",
          "zone_id": "us-west-2a",
          "spot_bid_price_percent": 100,
          "ebs_volume_count": 0
        },
        "node_type_id": "r6id.xlarge",
        "enable_elastic_disk": False,
        "data_security_mode": "SINGLE_USER",
        "runtime_engine": "STANDARD",
        "num_workers": 2
      }
    }
  ],
  "git_source": {
    "git_url": git_repo,
    "git_provider": git_provider,
    "git_branch": git_branch
  },
  "queue": {
    "enabled": True
  },
  "run_as": {
    "user_name": "merve.karali@databricks.com"
  }
}

job_config

# COMMAND ----------

import json
import requests
from requests.auth import HTTPBasicAuth

def create_databricks_job(job_config):
    # API endpoint
    url = f"{os.environ['DATABRICKS_HOST']}/api/2.0/jobs/create"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
    }
    
    # Authentication
    auth = HTTPBasicAuth("token",os.environ['DATABRICKS_TOKEN'])
    
    # Convert the jobs_config dictionary to a JSON string
    job_config_json = json.dumps(job_config)
    
    # Print the JSON content for verification (optional)
    print("Job configuration JSON:")
    print(job_config_json)
    
    try:
        # Make the API request
        response = requests.post(url, headers=headers, auth=auth, data=job_config_json)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Print the output
        print("Job created successfully:")
        print(json.dumps(result, indent=2))
        
        # Return the job ID
        return result.get('job_id')
    
    except requests.exceptions.RequestException as e:
        print(f"Error creating job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error details: {e.response.text}")
        return None



# COMMAND ----------

job_id = create_databricks_job(job_config)
if job_id:
    print(f"Created job with ID: {job_id}")
