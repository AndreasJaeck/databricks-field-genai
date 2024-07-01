# Databricks notebook source
# MAGIC %md
# MAGIC # Creating the chatbot with Retrieval Augmented Generation (RAG)
# MAGIC <br>
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-flow-2.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Prerequisite

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Create config 
# MAGIC
# MAGIC To keep track of different parameters and endpoints the Databricks Agent Framework leverages .yaml configuration files. This will make it much easier to swap an endpoint/model or use a different index. 

# COMMAND ----------

import mlflow
import yaml

# Set variables for config
config_name= "rag_chain_config"
catalog = "dbdemos_aj"
schema = "agent_framework"

# Define configuraion
rag_chain_config = {
    "databricks_resources": {
        "llm_endpoint_name": "databricks-dbrx-instruct",
        "catalog": f"{catalog}",
        "schema": f"{schema}",
        "model_name": "chain_review_app"
    },
    "input_example": {
        "messages": [{"content": "Databricks Lakehouse Platform", "role": "user"}]
    },
    "embedding_config": {
        "embedding_endpoint_name": "databricks-bge-large-en",
        "text_column": "content"
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 1500, "temperature": 0.01},
        "llm_prompt_template": "You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
        "llm_prompt_template_variables": ["context", "question"],
    },
    "retriever_config": {
        "vector_search_endpoint_name": "one-env-shared-endpoint-2",
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 5, "query_type": "ann"},
        "schema": {"chunk_text": "content", "document_uri": "url", "primary_key": "id"},
        "vector_search_index": f"{catalog}.{schema}.vs_index",
    },
}
try:
    with open(f'configs/{config_name}.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')

# COMMAND ----------

import os

VECTOR_SEARCH_ENDPOINT_NAME = model_config.get("vector_search_endpoint_name") 
vs_index_fullname = model_config.get("vs_index_name") 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Load config

# COMMAND ----------

import mlflow
model_config = mlflow.models.ModelConfig(development_config="configs/rag_chain_config.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create vector search chain object

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.vectorstores import DatabricksVectorSearch

embedding_model = DatabricksEmbeddings(endpoint=model_config.get("embedding_model"))
text_column = model_config.get("vs_index_text_column")

def get_retreiver(persist_dir = None):
  vsc = VectorSearchClient()
  vs_idx = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

  delta_vsc = DatabricksVectorSearch(vs_idx, text_column=text_column, embedding=embedding_model)
  return delta_vsc.as_retriever()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create chat object to connect to Llama endpoint

# COMMAND ----------

# Test Databricks Foundation LLM model
from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(
  endpoint=model_config.get("chat_model_name"), 
  max_tokens = model_config.get("chat_model_max_tokens")
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create prompt

# COMMAND ----------

from langchain.prompts import PromptTemplate

input_template = model_config.get("input_template")
TEMPLATE = input_template + "{context} Question: {question} Answer:"
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# COMMAND ----------

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)

retrieval = RunnableParallel(
    {"context": get_retreiver, "question": RunnablePassthrough()}
)

chain = retrieval | prompt | chat_model

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Set model for logging from lc_model

# COMMAND ----------

mlflow.models.set_model(model=chain)

# COMMAND ----------


