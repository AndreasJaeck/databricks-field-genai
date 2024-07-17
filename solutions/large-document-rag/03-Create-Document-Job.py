# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Excuting this notebook will create a job that will load .txt files from Volume, processes and syncs to the online table and vector index. 
# MAGIC
# MAGIC AJ: The Code that need's to be run is in 03b-Load-Documents

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway]
# MAGIC dbutils.library.restartPython()

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

# COMMAND ----------

import os
import time
from databricks.sdk.service import jobs


notebook_path = f'/Users/{w.current_user.me().user_name}/databricks-field-genai/solutions/large-document-rag/03b-Load-Documents'


cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")


# Defined a cron schedule to run every weekday at 6 PM UTC+2
cron_schedule = "0 0 18 * * ?"

created_job = w.jobs.create(name=f'databricks-field-genai-large-document-rag-{time.time_ns()}',
                            tasks=[
                                jobs.Task(description="test",
                                          existing_cluster_id=cluster_id,
                                          notebook_task=jobs.NotebookTask(notebook_path=notebook_path), #ToDO: Change to git provider
                                          task_key="test",
                                          timeout_seconds=0)
                            ],
                            schedule=jobs.CronSchedule(
                                quartz_cron_expression=cron_schedule,
                                timezone_id='Europe/Berlin'  # Set timezone to UTC+2
                            ) #ToDO: Provide another option for tigger start in volume source_data 
                            )

print(f"Created Job ID: {created_job.job_id}")

