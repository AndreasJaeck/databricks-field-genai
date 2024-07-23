# Databricks notebook source
# MAGIC %md
# MAGIC # 5/ Deploying our frontend App with Lakehouse Applications
# MAGIC
# MAGIC
# MAGIC Mosaic AI Agent Evaluation review app is used for collecting stakeholder feedback during your development process.
# MAGIC
# MAGIC You still need to deploy your own front end application!
# MAGIC
# MAGIC Let's leverage Databricks Lakehouse Applications to build and deploy our first, simple chatbot frontend app. 
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-frontend-app.png?raw=true" width="1200px">
# MAGIC
# MAGIC
# MAGIC *Note: Lakehouse apps are in preview, reach-out your Databricks Account team for more details.*
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F02-simple-app%2F03-Deploy-Frontend-Lakehouse-App&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F02-simple-app%2F03-Deploy-Frontend-Lakehouse-App&version=1">

# COMMAND ----------

# MAGIC %pip install --quiet -U mlflow databricks-sdk==0.23.0 gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,load lakehouse helpers
# MAGIC %run ./_resources/02-lakehouse-app-helpers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add your application configuration
# MAGIC
# MAGIC Lakehouse apps let you work with any python framework. For our small demo, we will create a small configuration file containing the model serving endpoint name used for our demo and save it in the `chatbot_app/app.yaml` file.

# COMMAND ----------

import os
import mlflow

#Get the configuration
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')
databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_config")
secrets_config = model_config.get("secrets_config")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Get Service Principal Token from secrets service 
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(secrets_config.get("secret_scope"), secrets_config.get("secret_key"))
# Get the host from the configuration
os.environ["DATABRICKS_HOST"] = databricks_resources.get("host")

# Assume role of Service Principal
w = WorkspaceClient(host=databricks_resources.get("host"), token=os.environ['DATABRICKS_TOKEN'])
sp_id = w.current_user.me().emails[0].value
print(f"Service Principal ID: {sp_id}")

# COMMAND ----------

import yaml
import os

# Name of the rag chain endpoint (Please update according to your endpoint name!)
endpoint_name = '<add your langchain endpoint name here!>'

host = databricks_resources.get("host")

endpoint_url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
os.environ["MODEL_SERVING_URL"] = endpoint_url

# Our frontend application will hit the model endpoint we deployed.
# Because dbdemos let you change your catalog and database, let's make sure we deploy the app with the proper endpoint name
yaml_app_config = {
    "command": ["uvicorn", "main:app", "--workers", "4"],
    "env": [
        {"name": "MODEL_SERVING_URL", "value": os.environ["MODEL_SERVING_URL"]},
        ],
}
try:
    with open("chatbot_app/app.yaml", "w") as f:
        yaml.dump(yaml_app_config, f)
except:
    print("pass to work on build job")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's now create our chatbot application using Gradio

# COMMAND ----------

# DBTITLE 1,Get filter values
from pyspark.sql.functions import col

document_table_schema = retriever_config.get("document_table_schema")

pdf = spark.read.table(databricks_resources.get("document_table")) \
    .select(col(document_table_schema.get('document_name_col')))\
    .toPandas()

filter_values = pdf[document_table_schema.get('document_name_col')].tolist()

# These are the values that you can pass to the Gradio app as filter values.
print(filter_values)

# COMMAND ----------

# MAGIC %%writefile chatbot_app/main.py
# MAGIC from fastapi import FastAPI
# MAGIC import gradio as gr
# MAGIC import os
# MAGIC import requests
# MAGIC from gradio.themes.utils import sizes
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC # Will automatically pick up credentials from env var's
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC app = FastAPI()
# MAGIC
# MAGIC def format_message(content, role, metadata_filter):
# MAGIC     """
# MAGIC     Format a message for the API request
# MAGIC     
# MAGIC     Args:
# MAGIC     content (str): The message content.
# MAGIC     role (str): The role of the message sender ('user' or 'assistant').
# MAGIC     metadata_filter (str): The filter to be applied to the message.
# MAGIC     
# MAGIC     Returns:
# MAGIC     dict: A formatted message dictionary.
# MAGIC     """
# MAGIC     message = {"content": content, "role": role}
# MAGIC     if metadata_filter and metadata_filter != "All Documents":
# MAGIC         message["filter"] = metadata_filter
# MAGIC     return message
# MAGIC
# MAGIC def chat(message, history, metadata_filter):
# MAGIC     """
# MAGIC     Process a chat message and return the updated conversation history.
# MAGIC     
# MAGIC     Args:
# MAGIC     message (str): The current user message.
# MAGIC     history (list): The conversation history.
# MAGIC     metadata_filter (str): The filter to be applied to the messages.
# MAGIC     
# MAGIC     Returns:
# MAGIC     list: The updated conversation history.
# MAGIC     """
# MAGIC     formatted_history = []
# MAGIC     for human, ai in history:
# MAGIC         formatted_history.extend([
# MAGIC             format_message(human, "user", metadata_filter),
# MAGIC             format_message(ai, "assistant", metadata_filter)
# MAGIC         ])
# MAGIC
# MAGIC     formatted_history.append(format_message(message, "user", metadata_filter))
# MAGIC
# MAGIC     try:
# MAGIC         response = requests.post(
# MAGIC             url=os.environ['MODEL_SERVING_URL'],
# MAGIC             headers={
# MAGIC                 'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
# MAGIC                 'Content-Type': 'application/json'
# MAGIC             },
# MAGIC             json={"messages": formatted_history}
# MAGIC         )
# MAGIC         response.raise_for_status()
# MAGIC         bot_message = response.json()["choices"][0]["message"]["content"]
# MAGIC     except Exception as e:
# MAGIC         bot_message = f"An error occurred: {str(e)}"
# MAGIC
# MAGIC     return history + [(message, bot_message)]
# MAGIC
# MAGIC theme = gr.themes.Soft(
# MAGIC     text_size=sizes.text_sm,
# MAGIC     radius_size=sizes.radius_sm,
# MAGIC     spacing_size=sizes.spacing_sm,
# MAGIC )
# MAGIC
# MAGIC # Add the filter values from document table here
# MAGIC filter_values = [
# MAGIC     "Travel-Cancellation-Insurance",
# MAGIC     "House-Insurance",
# MAGIC     "Car-Insurance",
# MAGIC ]
# MAGIC
# MAGIC with gr.Blocks(theme=theme) as demo:
# MAGIC     gr.Markdown(
# MAGIC         """
# MAGIC         # Chat with your Contract Copilot
# MAGIC         This chatbot is a demo example. <br>It answers with the help of Documentation saved in a Knowledge database.<br/>This content is provided as a LLM RAG educational example, without support.
# MAGIC         """
# MAGIC     )
# MAGIC
# MAGIC     gr.Markdown("### Chat History")
# MAGIC     chatbot = gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True)
# MAGIC     
# MAGIC     gr.Markdown("### Input")
# MAGIC     with gr.Row():
# MAGIC         with gr.Column(scale=7):
# MAGIC             msg = gr.Textbox(placeholder="What is an Insurance?", container=False, label="Your question")
# MAGIC         with gr.Column(scale=3):
# MAGIC             metadata_field = gr.Dropdown(label="Document Filter", choices=["All Documents"] + filter_values, value="All Documents")
# MAGIC
# MAGIC     def user(user_message, history, metadata_filter):
# MAGIC         return "", history + [[user_message, None]]
# MAGIC
# MAGIC     def bot(history, metadata_filter):
# MAGIC         if history and history[-1][1] is None:
# MAGIC             user_message = history[-1][0]
# MAGIC             bot_response = chat(user_message, history[:-1], metadata_filter)
# MAGIC             history[-1][1] = bot_response[-1][1]
# MAGIC         return history
# MAGIC
# MAGIC     msg.submit(user, [msg, chatbot, metadata_field], [msg, chatbot]).then(
# MAGIC         bot, [chatbot, metadata_field], chatbot
# MAGIC     )
# MAGIC
# MAGIC     gr.Button("Clear").click(lambda: None, None, chatbot, queue=False)
# MAGIC
# MAGIC     gr.Examples(
# MAGIC         examples=[
# MAGIC             ["What is an insurance for?", "begriffe"]
# MAGIC         ],
# MAGIC         inputs=[msg, metadata_field]
# MAGIC     )
# MAGIC
# MAGIC demo.queue(default_concurrency_limit=24)
# MAGIC app = gr.mount_gradio_app(app, demo, path="/")

# COMMAND ----------

demo.launch(share=True)

# COMMAND ----------

# Uncomment this line to run the app local from cluster. 
# It is recommended to add a password to your app! 
#demo.launch(share=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying our application to Lakehouse App's (Private Preview)
# MAGIC
# MAGIC Our application is made of 2 files under the `chatbot_app` folder:
# MAGIC - `main.py` containing our python code
# MAGIC - `app.yaml` containing our configuration
# MAGIC
# MAGIC All we now have to do is call the API to create a new app and deploy using the `chatbot_app` path:

# COMMAND ----------

helper = LakehouseAppHelper()
# Delete potential previous app version
helper.delete("large-doc-copilot")

# COMMAND ----------

#Helper is defined in the _resources/02-lakehouse-app-helpers notebook (temporary helper)

helper = LakehouseAppHelper()
#helper.list()

#Delete potential previous app version
#helper.delete("large-document-copilot")
helper.create("large-doc-copilot", app_description="Your Large Document Copilot")
helper.deploy("large-doc-copilot", os.path.join(os.getcwd(), 'chatbot_app'))

# COMMAND ----------

helper.list()


# COMMAND ----------

## Please make sure that the listed service_principal has the permission to query the chain endpoint! Othwise the front-end will not be able to query the model. 
helper.details("large-doc-copilot")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Your Lakehouse app is ready and deployed!
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-gradio-app.png?raw=true" width="750px" style="float: right; margin-left:10px">
# MAGIC
# MAGIC Open the UI to start requesting your chatbot.
# MAGIC
# MAGIC As improvement, we could improve our chatbot UI to provide feedback and send it to Mosaic AI Quality Labs, so that bad answers can be reviewed and improved.
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC We saw how Databricks provides an end to end platform: 
# MAGIC - Building and deploying an endpoint
# MAGIC - Buit-in solution to review, analyze and improve our chatbot
# MAGIC - Deploy front-end genAI application with lakehouse apps!
