# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Excuting this notebook will create a job that will load .txt files from Volume, processes and syncs to the online table and vector index.

# COMMAND ----------

import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()

notebook_path = f'/Users/{w.current_user.me().user_name}/databricks-field-genai/solutions/large-document-rag/03b-Load-Documents'
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

# Defined a cron schedule to run every weekday at 6 PM UTC+2
cron_schedule = "0 0 18 * * ?"

created_job = w.jobs.create(name=f'databricks-field-genai-large-document-rag-{time.time_ns()}',
                            tasks=[
                                jobs.Task(description="test",
                                          existing_cluster_id=cluster_id,
                                          notebook_task=jobs.NotebookTask(notebook_path=notebook_path),
                                          task_key="test",
                                          timeout_seconds=0)
                            ],
                            schedule=jobs.CronSchedule(
                                quartz_cron_expression=cron_schedule,
                                timezone_id='Europe/Berlin'  # Set timezone to UTC+2
                            )
                            )

print(f"Created Job ID: {created_job.job_id}")

# COMMAND ----------

# Optionally, run the job immediately
# run_by_id = w.jobs.run_now(job_id=created_job.job_id).result()

#print("job run finished successfully")
