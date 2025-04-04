# Databricks notebook source
# MAGIC %md
# MAGIC # Driver notebook
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export. We generated three notebooks in the same folder:
# MAGIC - [agent]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [**driver**]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to deploy the agent defined in the [agent]($./agent) notebook. The notebook does the following:
# MAGIC 1. Logs the agent to MLflow
# MAGIC 2. Evaluate the agent with Agent Evaluation
# MAGIC 3. Registers the agent to Unity Catalog
# MAGIC 4. Deploys the agent to a Model Serving endpoint
# MAGIC
# MAGIC ## Prerequisities
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent, the LLM endpoint, and the agent prompt.
# MAGIC - Review and run the [agent]($./agent) notebook in this folder to view the agent's code, iterate on the code, and test outputs.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow langchain==0.2.16 langgraph-checkpoint==1.0.12  langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic databricks_langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Get current username
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()

# Assemble experiment path
def get_experiment_name(postfix):
    return f"/Users/{username}/{postfix}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the [agent]($./agent) notebook. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow

experiment_name = get_experiment_name("clearstream-extraction-agent-experiment")
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

input_example = {
    "messages": [
        {
            "role": "user",
            "content": '"""{"Ticket ID": "1890051", "Service_Category": "New Issues", "Incident_Category": "Domestic - T2S", "Cause_Category": "Code eligibility", "Initial eMail": "Please set up isin FR0128465491 at clearstream, with the option to bridge to euroclear\nWe are C12345\nRegards"}"""'
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            'agent',
        ),
        pip_requirements=[
            "langchain==0.2.16",
            "langchain-community==0.2.16",
            "langgraph-checkpoint==1.0.12",
            "langgraph==0.2.16",
            "pydantic",
            "databricks_langchain", # used for the retriever tool
        ],
        model_config="config.yml",
        artifact_path='agent',
        input_example=input_example,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.

# COMMAND ----------

import pandas as pd


# An example for an evaluation dataset
eval_examples = [
    {
        "request": '"""{"Ticket ID": "1890051", "Service_Category": "New Issues", "Incident_Category": "Domestic - T2S", "Cause_Category": "Code eligibility", "Initial eMail": "Please set up isin FR0128465491 at clearstream, with the option to bridge to euroclear\nWe are C12345\nRegards"}"""',
        
        "expected_response": """```json\n{"ticket_id": "1890051","service_category": "New Issues", "incident_category": "Domestic - T2S", "cause_category": "Code eligibility", "customer_email": null, "isin": ["FR0128465491"], "account_number": ["C12345"], "record_date": null, "trade_date": null, "ex_date": null, "currency": null, "customer_first_name": null, "customer_last_name": null}\n```"""
    }
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)

# COMMAND ----------

import mlflow
import pandas as pd

# Specify evaluation guidelines 
global_guidelines = [
    "output must be a valid JSON string without any explanation text or additional fields",
    """Field validation rules if the field is present:
    - ticket_id: Must be null or a 7-digit number
    - service_category: Must be one of ["Cash", "Client Account", "Corporate Actions", "Income", "New Issues", "Settlement"]
    - incident_category: Free text field
    - cause_category: Free text field
    - customer_first_name: Free text field
    - customer_last_name: Free text field
    - customer_email: Must be valid email format
    - isin: Must be a list of valid ISIN codes (format: 2 letters + 9 alphanumeric + 1 check digit)
    - account_number: Must be a list
    - record_date: Must be in format DD-MM-YYYY
    - trade_date: Must be in format DD-MM-YYYY
    - ex_date: Must be in format DD-MM-YYYY
    - currency: Must be valid ISO 4217 3-letter code""",
    "Fields should only be included in the output if they are present and valid in the input"
]

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",  # replace `chain` with artifact_path that you used when calling log_model.
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
        evaluator_config={
            "databricks-agent": {
                "global_guidelines": global_guidelines
            }
        }
    )


# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "dbdemos_aj" # put your catalog name here
schema = "silver_ticket_system" # put your schema name here
model_name = "ticket_extraction"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})

# COMMAND ----------


