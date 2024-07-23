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

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway]
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
catalog = "<put catalog name here>" # adjust to use correct catalog/ will be generated if not exist
schema = "<put database name here>" # adjust to write to the correct db / will be generated if not exist


# Endpoint Names
# ---------------------------
vector_search_endpoint_name = "<put vector search endpoint here>"# Please create Vector Search Endpoint through UI if no Endpoint exists. 
llm_endpoint_name = "<name of the llm serving endpoint>"# Pay-per-token endpoints are probabaly the easiest to start with. If you don't have access you can integrate an external endpoint (like OpenAI) or host your own LLM with Throughput Serving. Check on Serving tab (left bottom corner) which Endpoints are available. 
embedding_endpoint_name = "<name of the embedding model endpoint>"#  Name of the embedding model. Powerful models are available on Databricks Marketplace and can get deployed as a serving endpoint with less hardware requirements compared to LLM-serving. 


# Service Principal Secret Params
#----------------------------
secret_scope = "rag-sp"
secret_key = "rag_sp_token"

# Rag Chain Model Name
#----------------------------
rag_chain_model_name = "chain_model"

# Doc Splitting
# ---------------------------
# Using 8k tokens for parent chunk size, since LLM attention is decreasing quickyly with larger windows.
n_child_tokens = 1024 
n_parent_tokens = n_child_tokens * 8


# Table Schema
# ---------------------------
# (please adjust only if really necessary)

text_col = "content" # col that contains the actual natural language/text
document_uri = "url" # the path to the document (can be a path to a url)
document_name_col = "document_name" # col that contain the name of the document (used for filtering)
document_id_col = "document_id" # col that contain the unique id of the document
parent_document_id_col = "parent_split_id" # col that contain the unique id of the parent document
child_document_id_col = "child_split_id" # col that contain the unique id of the child document
timestamp_col= "timestamp" # name of timestamp column

# ---------------------------
# (please adjust only if really necessary)

document_table_posfix = "documents"
document_feature_spec = "parent_document_spec"


# COMMAND ----------

# DBTITLE 1,Configuration .yaml
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
        "rag_chain_model_name": rag_chain_model_name,
        "source_volume": "source_data",
        "checkpoint_volume": "checkpoints",
        "source_volume_path": f"/Volumes/{catalog}/{schema}/source_data/text",
        "checkpoint_volume_path": f"/Volumes/{catalog}/{schema}/checkpoints",
        "document_table": f"{catalog}.{schema}.{document_table_posfix}",
        "parent_table": f"{catalog}.{schema}.{document_table_posfix}_parent_splits",
        "child_table": f"{catalog}.{schema}.{document_table_posfix}_child_splits",
        "vector_search_index": f"{catalog}.{schema}.{document_table_posfix}_child_splits_index",
        "document_feature_spec_uri": f"{catalog}.{schema}.{document_feature_spec}",
        "document_feature_spec": document_feature_spec,
        "host": "https://" + spark.conf.get("spark.databricks.workspaceUrl"),
    },
    "secrets_config": {"secret_scope": secret_scope, "secret_key": secret_key},
    "input_example": {
        "messages": [
            {
                "content": "What are the conditions of my insurance?",
                "role": "user",
                "filter": "All Documents",
            },
            {
                "content": "The conditions can vary based on the specic type of insurance",
                "role": "assistant",
                "filter": "All Documents",
            },
            {
                "content": "Tell me about the travel cancellation insurance.",
                "role": "user",
                "filter": "All Documents",
            },
        ]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 3000, "temperature": 0.01},
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
        "parameters": {"k": 3},
        "embedding_dimension": 1024,
    },
}
try:
    with open(f"{config_name}.yaml", "w") as f:
        yaml.dump(rag_chain_config, f)
except:
    print("pass to work on build job")
