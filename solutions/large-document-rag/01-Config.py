# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Setup
# MAGIC
# MAGIC We leverage the Databricks Agent Framework to build our RAG Application in a more configuration driven way and allow for more advanced review and feedback workflows. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Install Dependencies 

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Create Configuration

# COMMAND ----------

# DBTITLE 1,Configurations
import mlflow
import yaml

## Set configuration variables
config_name= "rag_chain_config"

# Table/Volume Names 
# ---------------------------
catalog = "dbdemos_aj" # adjust to use correct catalog
schema = "dbdemos_large_document_rag" # adjust to create unique db 


# Endpoint Names
# ---------------------------
vector_search_endpoint_name = "one-env-shared-endpoint-2"
llm_endpoint_name = "databricks-dbrx-instruct"
embedding_endpoint_name = "databricks-bge-large-en"

# Doc Splitting
# ---------------------------
# Using 8k tokens for parent chunk size, since LLM attention is decreasing quickyly with larger windows.
n_child_tokens = 1024
n_parent_tokens = n_child_tokens * 8

# Table Schema (please adjust only if really necessary)
# ---------------------------
text_col = "content"
document_uri = "url"
document_name_col = "document_name"
document_id_col = "document_id"
parent_document_id_col = "parent_split_id"
child_document_id_col = "child_split_id"
timestamp_col= "timestamp"


# COMMAND ----------

# DBTITLE 1,Configuraion .yaml
# Define configuraion
rag_chain_config = {
    "databricks_resources": {
        "catalog": catalog,
        "schema": schema,
        "llm_endpoint_name": llm_endpoint_name,
        "vector_search_endpoint_name": vector_search_endpoint_name,
        "embedding_endpoint_name": embedding_endpoint_name,
        "feature_serving_endpoint_name": f"{schema}-parent-splits",
        "rag_chain_endpoint_name": f"{schema}-chain",
        "source_volume": "source_data",
        "checkpoint_volume": "checkpoints",
        "source_volume_path": f"/Volumes/{catalog}/{schema}/source_data/text",
        "checkpoint_volume_path": f"/Volumes/{catalog}/{schema}/checkpoints",
        "document_table": f"{catalog}.{schema}.documents",
        "parent_table": f"{catalog}.{schema}.documents_parent_splits",
        "child_table": f"{catalog}.{schema}.documents_child_splits",
        "vector_search_index": f"{catalog}.{schema}.documents_child_splits_index",
        "host": "https://" + spark.conf.get("spark.databricks.workspaceUrl"),
    },
    "secrets_config": {"secret_scope": "dbdemos", "secret_key": "rag_sp_token"},
    "input_example": {
        "messages": [
            {"content": "What is Databricks Lakehouse Platform?", "role": "user"}
        ]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 1500, "temperature": 0.01},
        "llm_prompt_template": "You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
        "llm_prompt_template_variables": ["context", "question"],
    },
    "retriever_config": {
        "n_parent_tokens": n_parent_tokens,
        "n_child_tokens": n_child_tokens,
        "document_table_schema": {
            "text_col": text_col,
            "document_uri": document_uri,
            "timestamp_col": timestamp_col,
            "document_name_col": document_name_col,
            "primary_key": document_id_col,
        },
        "parent_splits_table_schema": {
            "text_col": text_col,
            "document_uri": document_uri,
            "timestamp_col": timestamp_col,
            "document_name_col": document_name_col,
            "parent_split_index": "parent_split_index",
            "document_foreign_key": document_id_col,
            "primary_key": parent_document_id_col,
        },
        "child_splits_table_schema": {
            "text_col": text_col,
            "embedding_vector_col": "embedding_vector",
            "document_uri": document_uri,
            "timestamp_col": timestamp_col,
            "document_name_col": document_name_col,
            "parent_split_index": "parent_split_index",
            "child_split_index": "child_split_index",
            "document_foreign_key": document_id_col,
            "parent_split_foreign_key": parent_document_id_col,
            "primary_key": child_document_id_col,
        },
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 5, "query_type": "ann"},
        "embedding_dimension": 1024 
    },
}
try:
    with open(f"config/{config_name}.yaml", "w") as f:
        yaml.dump(rag_chain_config, f)
except:
    print("pass to work on build job")
