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

experiment_name = get_experiment_name("clearstream-categorization-agent-experiment")
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Dear All, We do not recognize below cash entry (Currency/amount and Value date). Please provide us with the necessary documentation in order to assign this cash amount to the correct processing unit. Please note, apart from the ISIN we would like to receive the following: • Evidences which show that this cash entry concerns either a Corporate Action-/Income event or is related to a specific Trade • Any Contract Notes, Trade Confirmations, Notifications or relevant Transaction Breakdowns which the Counterparty is able to produce. • Our references (e.g. GZUXXXXXXXXXXXX) which ties this amount to a specific transaction already confirmed to us."
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
        "request": "Dear All, We do not recognize below cash entry (Currency/amount and Value date). Please provide us with the necessary documentation in order to assign this cash amount to the correct processing unit. Please note, apart from the ISIN we would like to receive the following: • Evidences which show that this cash entry concerns either a Corporate Action-/Income event or is related to a specific Trade • Any Contract Notes, Trade Confirmations, Notifications or relevant Transaction Breakdowns which the Counterparty is able to produce. • Our references (e.g. GZUXXXXXXXXXXXX) which ties this amount to a specific transaction already confirmed to us.",
        
        "expected_response": '```json\n{\n  "service_category": "Cash",\n  "incident_category": "Suspense",\n  "cause_category": "Current Status",\n  "service_category_reasoning": "The query is related to a cash entry that is not recognized, which falls under the Cash service category.",\n  "incident_category_reasoning": "The incident is related to a suspense situation where the cash entry is not recognized, which falls under the Suspense incident category.",\n  "cause_category_reasoning": "The cause of the incident is related to the current status of the cash entry, which is not recognized, and requires documentation to assign it to the correct processing unit, which falls under the Current Status cause category."\n}\n```'
    },
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)

# COMMAND ----------

import mlflow
import pandas as pd

# Specify evaluation guidelines 
# Note AJ: Here we check if required keys are present and if categories are applied to the correct format
global_guidelines = [
    "output json must be a valid json string, output json must contain the following keys: service_category, incident_category, cause_category, service_category_reasoning, incident_category_reasoning, cause_category_reasoning",
    """ Each category label must be one of the following: {
    "service_categories": [
        "Cash",
        "Corporate Action",
        "Income",
        "Settlement",
        "New Issues",
        "Client Account",
    ],
    "category_tree": {
        "Cash": {
            "New Instruction": ["Deadline", "Formats"],
            "Suspense": ["Current Status", "Forcing Requests", "Late instruction"],
            "Final": ["Cancellation/Rejection", "Instruction Details"],
            "General": [
                "Account credit facilities",
                "Compliance",
                "Currency specifics",
            ],
        },
        "Corporate Action": {
            "Pre-notification": ["Event Existence", "Instruction at holder initiative"],
            "Notification sent (MT564)": [
                "Client Instruction",
                "Compensation/Transformation",
                "Event Details",
                "Proceeds/Payment",
            ],
            "Confirmation sent (MT566)": [
                "Compensation/Transformation",
                "Payment Details",
            ],
        },
        "Income": {
            "Pre-notification": ["Event Existence"],
            "Notification sent (MT564)": [
                "Compensation/Transformation",
                "Event Details",
                "Proceeds/Payment",
            ],
            "Confirmation sent (MT566)": [
                "Compensation/Transformation",
                "Payment Details",
                "Reversal",
            ],
        },
        "Settlement": {
            "New instruction": ["Allegement", "Formats"],
            "Suspense": ["Allegement", "Cancellation request", "Current status"],
            "Final": ["Cancellation/Rejection", "Instruction Details"],
            "General": [
                "Compliance",
                "Foreign Transaction Tax",
                "Penalties",
                "Market specifications",
            ],
        },
        "New Issues": {
            "International": [
                "Code eligibility",
                "Financial Instruments static data",
                "Syndicated Distribution",
            ],
            "Domestic - T2S": ["Code eligibility", "Financial Instruments static data"],
            "Domestic - outside T2S": [
                "Code eligibility",
                "Financial Instruments static data",
            ],
        },
        "Client Account": {
            "Account Information": [
                "Account settings",
                "Audit Request",
                "Cash Balance",
                "Client Claim",
                "Client Contingency",
                "Securities Balance",
                "Historical Research",
                "Interest Rates",
                "Reoccuring Reports",
                "Statement Request",
            ],
            "Billing": ["Billing details", "Billing Portal", "Invoice copy"],
        },
    },
}
"""
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
catalog = "dbdemos_aj"
schema = "silver_ticket_system"
model_name = "ticket_categorization"
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


