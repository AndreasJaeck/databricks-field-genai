# Databricks notebook source
# MAGIC %md 
# MAGIC # Run configured chain, track and evaluate, trace
# MAGIC - Create chain with different configurations -> fast and secure experimentation
# MAGIC - Log models and configurations
# MAGIC - Use Databricks fine-tuned LLMs for LLM as a judge
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-llm-as-a-judge.png?raw=true" style="float: right" width="900px">

# COMMAND ----------

# MAGIC %md ### Pre-requisit - run on personal compute

# COMMAND ----------

# MAGIC %pip install databricks-rag-studio "mlflow@git+https://github.com/mlflow/mlflow.git@databricks-rag-studio"
# MAGIC
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md ### Define paths to chain notebook and configs

# COMMAND ----------

import os 

dir_path = os.getcwd()
chain_notebook_path = os.path.join(dir_path, "0 Create RAG Chain Config Driven")
config_dir = os.path.join(dir_path, "configs")

# COMMAND ----------

# MAGIC %md ### Read evaluation data

# COMMAND ----------

eval_data = spark.read.table("alan_demos.genai.evaluation_dataset")
eval_data = eval_data.selectExpr(
  "string(id) as request_id", 
  'question as request', 
  'answer as expected_response'
  )
eval_data = eval_data.limit(10).toPandas()
display(eval_data)

# COMMAND ----------

# MAGIC %md ### Run chain for each configuartion, log and evaluate model

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow

for conf in ["dbrx.yaml", "llama2.yaml", "llama3.yaml"]:
    config_path = os.path.join(config_dir, conf)

    with mlflow.start_run(run_name=conf):
        # Log the chain code + config + parameters to the run
        logged_chain_info = mlflow.langchain.log_model(
            lc_model=chain_notebook_path,
            model_config=config_path, 
            artifact_path="chain",
            input_example="What is Spark",
            example_no_conversion=True,  # required to allow the schema to work
            extra_pip_requirements=[
                "databricks-rag-studio==0.2.0"
            ],
        )

        chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
        eval_data["predictions"] = eval_data["request"].apply(lambda x: chain.invoke(x).content)

        eval_results = mlflow.evaluate(
            data=eval_data,
            predictions="predictions",
            targets="expected_response",
            model_type="databricks-rag",
        )

# COMMAND ----------

# MAGIC %md ### Trace Chains

# COMMAND ----------

mlflow.langchain.autolog(disable=False)

# COMMAND ----------

chain = mlflow.langchain.load_model(logged_chain_info.model_uri)

chain.invoke("How to track billing?").content
print("")

# COMMAND ----------


