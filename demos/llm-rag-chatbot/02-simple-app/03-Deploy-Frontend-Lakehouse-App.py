# Databricks notebook source
# MAGIC %md
# MAGIC # 3/ Deploying our frontend App with Lakehouse Applications
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

# MAGIC %pip install --quiet -U mlflow databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,load lakehouse helpers
# MAGIC %run ../_resources/02-lakehouse-app-helpers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add your application configuration
# MAGIC
# MAGIC Lakehouse apps let you work with any python framework. For our small demo, we will create a small configuration file containing the model serving endpoint name used for our demo and save it in the `chatbot_app/app.yaml` file.

# COMMAND ----------

MODEL_NAME = "dbdemos_rag_demo"
endpoint_name = f'agents_{catalog}-{db}-{MODEL_NAME}'[:63]

# Our frontend application will hit the model endpoint we deployed.
# Because dbdemos let you change your catalog and database, let's make sure we deploy the app with the proper endpoint name
yaml_app_config = {"command": ["uvicorn", "main:app", "--workers", "4"],
                    "env": [{"name": "MODEL_SERVING_ENDPOINT", "value": endpoint_name}]
                  }
try:
    with open('chatbot_app/app.yaml', 'w') as f:
        yaml.dump(yaml_app_config, f)
except:
    print('pass to work on build job')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's now create or chatbot application using Gradio

# COMMAND ----------

# MAGIC %%writefile chatbot_app/main.py
# MAGIC import itertools
# MAGIC from fastapi import FastAPI
# MAGIC import gradio as gr
# MAGIC import os
# MAGIC from gradio.themes.utils import sizes
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
# MAGIC
# MAGIC app = FastAPI()
# MAGIC
# MAGIC #your endpoint will directly be setup with proper permissions when you deploy your app
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC model_serving_endpoint_name = os.environ['MODEL_SERVING_ENDPOINT']
# MAGIC
# MAGIC def respond(message, history):
# MAGIC     if len(message.strip()) == 0:
# MAGIC         return "ERROR the question should not be empty"
# MAGIC     try:
# MAGIC         #TODO: we should send the history too - PR welcome!
# MAGIC         response = w.serving_endpoints.query(
# MAGIC             name=model_serving_endpoint_name,
# MAGIC             messages=[ChatMessage(content=message, role=ChatMessageRole.USER)],
# MAGIC             temperature=1.0,
# MAGIC             stream=False
# MAGIC         )
# MAGIC     except Exception as error:
# MAGIC         return f"ERROR requesting endpoint {model_serving_endpoint_name}: {error}"       
# MAGIC     return response.choices[0].message.content
# MAGIC
# MAGIC theme = gr.themes.Soft(
# MAGIC     text_size=sizes.text_sm,radius_size=sizes.radius_sm, spacing_size=sizes.spacing_sm,
# MAGIC )
# MAGIC
# MAGIC demo = gr.ChatInterface(
# MAGIC     respond,
# MAGIC     chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
# MAGIC     textbox=gr.Textbox(placeholder="What is RAG?",
# MAGIC                        container=False, scale=7),
# MAGIC     title="Databricks App RAG demo - Chat with your Databricks assistant",
# MAGIC     description="This chatbot is a demo example for the dbdemos llm chatbot. <br>It answers with the help of Databricks Documentation saved in a Knowledge databse.<br/>This content is provided as a LLM RAG educational example, without support. It is using DBRX, can hallucinate and should not be used as production content.<br>Please review our dbdemos license and terms for more details.",
# MAGIC     examples=[["What is DBRX?"],
# MAGIC               ["How can I start a Databricks cluster?"],
# MAGIC               ["What is a Databricks Cluster Policy?"],
# MAGIC               ["How can I track billing usage on my workspaces?"],],
# MAGIC     cache_examples=False,
# MAGIC     theme=theme,
# MAGIC     retry_btn=None,
# MAGIC     undo_btn=None,
# MAGIC     clear_btn="Clear",
# MAGIC )
# MAGIC
# MAGIC demo.queue(default_concurrency_limit=100)
# MAGIC app = gr.mount_gradio_app(app, demo, path="/")
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying our application
# MAGIC
# MAGIC Our application is made of 2 files under the `chatbot_app` folder:
# MAGIC - `main.py` containing our python code
# MAGIC - `app.yaml` containing our configuration
# MAGIC
# MAGIC All we now have to do is call the API to create a new app and deploy using the `chatbot_app` path:

# COMMAND ----------

#Helper is defined in the _resources/02-lakehouse-app-helpers notebook (temporary helper)
helper = LakehouseAppHelper()
#helper.list()
#Delete potential previous app version
#helper.delete("dbdemos-rag-chatbot-app")
helper.create("dbdemos-rag-chatbot-app", app_description="Your Databricks assistant")
helper.deploy("dbdemos-rag-chatbot-app", os.path.join(os.getcwd(), 'chatbot_app'))

# COMMAND ----------

helper.list()

# COMMAND ----------

helper.details("dbdemos-rag-chatbot-app")

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: ready to take it to a next level?
# MAGIC
# MAGIC Open the [03-advanced-app/01-PDF-Advanced-Data-Preparation]($../03-advanced-app/01-PDF-Advanced-Data-Preparation) notebook series to learn more about unstructured data, advanced chain, model evaluation and monitoring.
