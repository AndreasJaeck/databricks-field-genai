# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create Endpoints

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##Install Dependencies 

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents mlflow-skinny mlflow mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch databricks-sdk==0.23.0
# MAGIC %pip install databricks-feature-engineering databricks-sql-connector --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Get configuration

# COMMAND ----------

import os
import mlflow

#Get the conf from the local conf file
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

databricks_resources = model_config.get("databricks_resources")
secrets_config = model_config.get("secrets_config")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Set variables and permissions, create db's

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
    indexes = vsc.list_indexes(endpoint_name).get("vector_indexes", list())
    if any(index_full_name == index.get("name") for index in indexes):
      return True
    #Temp fix when index is not available in the list
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready')
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Get Service Principal Token from secrets service 
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(secrets_config.get("secret_scope"), secrets_config.get("secret_key"))
# Get the host from the configuration
os.environ["DATABRICKS_HOST"] = databricks_resources.get("host")

w = WorkspaceClient(host=databricks_resources.get("host"), token=os.environ['DATABRICKS_TOKEN'])
sp_id = w.current_user.me().emails[0].value
print(f"Service Principal ID: {sp_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Feature Serving Endpoint

# COMMAND ----------

# DBTITLE 1,Create Feature Serving Endpoint
# Import necessary classes
from databricks.feature_engineering import FeatureFunction, FeatureLookup, FeatureEngineeringClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk import WorkspaceClient


# Create a feature store client
fe = FeatureEngineeringClient()

## Create Feature Serving Endpoint

# Get configuration parameters
parent_splits_table_schema = retriever_config.get("parent_splits_table_schema")

# Create a lookup to fetch features by key
features=[
  FeatureLookup(
    table_name=databricks_resources.get("parent_table"),
    lookup_key=parent_splits_table_schema.get("primary_key"),
    feature_names=[parent_splits_table_schema.get("text_col"), parent_splits_table_schema.get("document_name_col"), parent_splits_table_schema.get("document_uri")]
  )
]

try:
  fe.create_feature_spec(name=databricks_resources.get('document_feature_spec_uri'), features=features, exclude_columns=None)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# Create endpoint for serving parent documents
try:
  endpoint_config = EndpointCoreConfigInput(
    name=databricks_resources.get("feature_serving_endpoint_name"), 
    served_entities=[
       ServedEntityInput(
          entity_name=databricks_resources.get('document_feature_spec_uri'), 
          workload_size="Small", 
          scale_to_zero_enabled=False,  
          environment_vars={
                 "DATABRICKS_TOKEN": f"{{secrets/{secrets_config.get('secret_scope')}/{secrets_config.get('secret_key')}}}",
                 "DATABRICKS_HOST": f"{databricks_resources.get('host')}"
          }
        )
      ]
    )
  
  status = w.serving_endpoints.create_and_wait(name=databricks_resources.get("feature_serving_endpoint_name"), config=endpoint_config)

  # Print endpoint creation status
  print(status)

except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Build Retriever

# COMMAND ----------

# MAGIC %%writefile src/utils.py
# MAGIC
# MAGIC import os
# MAGIC import requests
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain_community.embeddings import DatabricksEmbeddings
# MAGIC from langchain_core.retrievers import BaseRetriever
# MAGIC from langchain.schema import Document
# MAGIC from langchain.callbacks.manager import CallbackManagerForRetrieverRun
# MAGIC from typing import List
# MAGIC from urllib.parse import quote
# MAGIC
# MAGIC
# MAGIC class DatabricksParentDocumentRetriever(BaseRetriever):
# MAGIC     """
# MAGIC     A class for retrieving relevant parent documents based on a query using Databricks Vector Search and Online Table Serving.
# MAGIC
# MAGIC     Args:
# MAGIC         vs_index (DatabricksVectorSearch): The Databricks Vector Search index object.
# MAGIC         embedding_model (DatabricksEmbeddings): The Databricks Embeddings model for generating query embeddings.
# MAGIC         deploy_client (object): The deployment client for querying the feature store.
# MAGIC         feature_endpoint_name (str): The name of the feature serving endpoint.
# MAGIC         parent_id_key (str): The key for the parent document ID in the feature store.
# MAGIC         content_col (str): The column name for the parent document content in the feature store.
# MAGIC         filter_col (str): The column name for the filter value in the feature store.
# MAGIC         source_col (str): The column name for the source information in the feature store.
# MAGIC
# MAGIC     Attributes:
# MAGIC         vs_index (DatabricksVectorSearch): The Databricks Vector Search index object.
# MAGIC         embedding_model (DatabricksEmbeddings): The Databricks Embeddings model for generating query embeddings.
# MAGIC         deploy_client (object): The deployment client for querying the feature store.
# MAGIC         feature_endpoint_name (str): The name of the feature serving endpoint.
# MAGIC         parent_id_key (str): The key for the parent document ID in the feature store.
# MAGIC         content_col (str): The column name for the parent document content in the feature store.
# MAGIC         filter_col (str): The column name for the filter value in the feature store.
# MAGIC         source_col (str): The column name for the source information in the feature store.
# MAGIC     """
# MAGIC
# MAGIC     vs_index: object
# MAGIC     embedding_model: object
# MAGIC     deploy_client: object
# MAGIC     feature_endpoint_name: str
# MAGIC     parent_id_key: str
# MAGIC     content_col: str
# MAGIC     filter_col: str
# MAGIC     source_col: str
# MAGIC
# MAGIC     def _get_relevant_documents(
# MAGIC         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
# MAGIC     ) -> List[Document]:
# MAGIC         """
# MAGIC         Retrieves relevant parent documents based on a query.
# MAGIC
# MAGIC         Args:
# MAGIC             query (str or dict): The query string or a dictionary containing the question and optional filter value.
# MAGIC             run_manager (CallbackManagerForRetrieverRun): The run manager for managing callbacks during retrieval.
# MAGIC
# MAGIC         Returns:
# MAGIC             List[Document]: A list of relevant parent documents.
# MAGIC         """
# MAGIC         
# MAGIC         # Generate embedding vector from query content
# MAGIC         if "content" in query:
# MAGIC             embedding_vector = self.embedding_model.embed_query(query["content"])
# MAGIC         else:
# MAGIC             embedding_vector = self.embedding_model.embed_query(query)
# MAGIC
# MAGIC         # Set filter
# MAGIC         if "filter" in query: 
# MAGIC             filters = {self.filter_col: query["filter"]}
# MAGIC         else:
# MAGIC             filters = None
# MAGIC
# MAGIC         # Perform similarity search in the vector index to find matching parent document IDs
# MAGIC         resp = self.vs_index.similarity_search(
# MAGIC             columns=[self.parent_id_key],
# MAGIC             query_vector=embedding_vector,
# MAGIC             num_results=3,
# MAGIC             filters=filters,
# MAGIC         )
# MAGIC
# MAGIC         data = resp.get("result", {}).get("data_array", None)
# MAGIC
# MAGIC         # Handle case where no matching documents are found
# MAGIC         if not data:
# MAGIC             result_docs = [Document("no context found")]
# MAGIC         else:
# MAGIC             # Create unique set of IDs so we are not retrieving the same document twice
# MAGIC             parent_document_ids = list(
# MAGIC                 set([int(document_id) for document_id, distance in data])
# MAGIC             )
# MAGIC
# MAGIC             ## Get parent documents with parent IDs
# MAGIC             # Put IDs into TF format to query the feature serving endpoint
# MAGIC             ids = {
# MAGIC                 "dataframe_records": [
# MAGIC                     {self.parent_id_key: id} for id in parent_document_ids
# MAGIC                 ]
# MAGIC             }
# MAGIC
# MAGIC             # Query the feature serving endpoint to retrieve parent document content
# MAGIC             parent_content = self.deploy_client.predict(endpoint=self.feature_endpoint_name, inputs=ids)
# MAGIC     
# MAGIC             # Convert retrieved content into Document objects
# MAGIC             result_docs = [
# MAGIC                 Document(
# MAGIC                     page_content=doc[self.content_col],
# MAGIC                     metadata={
# MAGIC                         self.source_col: doc[self.source_col],
# MAGIC                         self.filter_col: doc[self.filter_col],
# MAGIC                         self.parent_id_key: doc[self.parent_id_key],
# MAGIC                     },
# MAGIC                 )
# MAGIC                 for doc in parent_content["outputs"]
# MAGIC             ]
# MAGIC
# MAGIC         return result_docs

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Q&A Chain
# MAGIC
# MAGIC If you want to use a simple Q&A chain use the code below. If you need a full conversational chain jump to conversational chain chapter.

# COMMAND ----------

# MAGIC %%writefile src/chain.py
# MAGIC
# MAGIC import os
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import mlflow
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC from operator import itemgetter
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import PromptTemplate
# MAGIC from langchain_core.runnables import RunnablePassthrough
# MAGIC from langchain_community.embeddings import DatabricksEmbeddings
# MAGIC from src.utils import DatabricksParentDocumentRetriever
# MAGIC
# MAGIC
# MAGIC # Workspace client will automatically pick up credentials from the environment var 'DATABRICKS_HOST' and 'DATABRICKS_TOKEN'
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC #Get the configuration from the local config file
# MAGIC model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC secrets_config = model_config.get("secrets_config")
# MAGIC retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_retrieval_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]
# MAGIC       
# MAGIC ## Get Vector Index
# MAGIC vsc = VectorSearchClient(
# MAGIC     disable_notice=True, 
# MAGIC     workspace_url=os.environ["DATABRICKS_HOST"],
# MAGIC     personal_access_token=os.environ["DATABRICKS_TOKEN"],
# MAGIC     )
# MAGIC
# MAGIC vs_index = vsc.get_index(
# MAGIC     endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
# MAGIC     index_name=databricks_resources.get("vector_search_index"),
# MAGIC )
# MAGIC
# MAGIC ## Get embedding endpoint
# MAGIC embedding_model = DatabricksEmbeddings(
# MAGIC     endpoint=databricks_resources.get("embedding_endpoint_name")
# MAGIC )
# MAGIC
# MAGIC # Get deployment client
# MAGIC deploy_client = get_deploy_client("databricks")
# MAGIC
# MAGIC # Get parent configuration
# MAGIC parent_splits_table_schema = retriever_config.get("parent_splits_table_schema")
# MAGIC
# MAGIC # Create custom LangChain retriever
# MAGIC retriever = DatabricksParentDocumentRetriever(
# MAGIC     vs_index=vs_index,
# MAGIC     embedding_model=embedding_model,
# MAGIC     deploy_client=deploy_client,
# MAGIC     feature_endpoint_name=databricks_resources.get("feature_serving_endpoint_name"), 
# MAGIC     parent_id_key=parent_splits_table_schema.get("primary_key"),
# MAGIC     content_col=parent_splits_table_schema.get("text_col"),
# MAGIC     filter_col=parent_splits_table_schema.get("document_name_col"),
# MAGIC     source_col=parent_splits_table_schema.get("document_uri")
# MAGIC )
# MAGIC
# MAGIC # Required to:
# MAGIC # 1. Enable the Agent Framework Review App to properly display retrieved chunks
# MAGIC # 2. Enable evaluation suite to measure the retriever
# MAGIC mlflow.models.set_retriever_schema(
# MAGIC     primary_key=parent_splits_table_schema.get("primary_key"),
# MAGIC     text_column=parent_splits_table_schema.get("text_col"),
# MAGIC     doc_uri=parent_splits_table_schema.get("document_uri"),
# MAGIC     other_columns=[
# MAGIC         parent_splits_table_schema.get("document_name_col"),
# MAGIC         parent_splits_table_schema.get("primary_key"),
# MAGIC     ],
# MAGIC )
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt
# MAGIC def format_context(docs):
# MAGIC     chunk_template = retriever_config.get("chunk_template")
# MAGIC     chunk_contents = [
# MAGIC         chunk_template.format(
# MAGIC             chunk_text=d.page_content,
# MAGIC         )
# MAGIC         for d in docs
# MAGIC     ]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC
# MAGIC # Prompt Template for generation
# MAGIC prompt = PromptTemplate(
# MAGIC     template=llm_config.get("llm_prompt_template"),
# MAGIC     input_variables=llm_config.get("llm_prompt_template_variables"),
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # FM for generation
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC     extra_params=llm_config.get("llm_parameters"),
# MAGIC )
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages")
# MAGIC         | RunnableLambda(extract_user_query_string),
# MAGIC         "context": itemgetter("messages")
# MAGIC         | RunnableLambda(extract_user_retrieval_string)
# MAGIC         | retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Conversational Chain

# COMMAND ----------

# MAGIC %%writefile src/conversational_chain.py
# MAGIC
# MAGIC import os
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import mlflow
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC from operator import itemgetter
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC from langchain_community.chat_models import ChatDatabricks
# MAGIC from langchain_community.vectorstores import DatabricksVectorSearch
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import (
# MAGIC     PromptTemplate,
# MAGIC     ChatPromptTemplate,
# MAGIC     MessagesPlaceholder,
# MAGIC )
# MAGIC from langchain_core.runnables import RunnablePassthrough, RunnableBranch
# MAGIC from langchain_core.messages import HumanMessage, AIMessage
# MAGIC from langchain_community.embeddings import DatabricksEmbeddings
# MAGIC from src.utils import DatabricksParentDocumentRetriever
# MAGIC
# MAGIC
# MAGIC # Workspace client will automatically pick up credentials from the environment var 'DATABRICKS_HOST' and 'DATABRICKS_TOKEN'
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC #Get the configuration from the local config file
# MAGIC model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC secrets_config = model_config.get("secrets_config")
# MAGIC retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC # Return the string filter value of the most recent message from the user
# MAGIC def extract_filter_value(chat_messages_array):
# MAGIC     if "filter" in chat_messages_array[-1]:
# MAGIC         return chat_messages_array[-1]["filter"]
# MAGIC     else:
# MAGIC         return None
# MAGIC     
# MAGIC     
# MAGIC # Return the chat history, which is is everything before the last question
# MAGIC def extract_chat_history(chat_messages_array):
# MAGIC     return chat_messages_array[:-1]
# MAGIC       
# MAGIC ## Get Vector Index
# MAGIC vsc = VectorSearchClient(
# MAGIC     disable_notice=True, 
# MAGIC     workspace_url=os.environ["DATABRICKS_HOST"],
# MAGIC     personal_access_token=os.environ["DATABRICKS_TOKEN"],
# MAGIC     )
# MAGIC
# MAGIC vs_index = vsc.get_index(
# MAGIC     endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
# MAGIC     index_name=databricks_resources.get("vector_search_index"),
# MAGIC )
# MAGIC
# MAGIC ## Get embedding endpoint
# MAGIC embedding_model = DatabricksEmbeddings(
# MAGIC     endpoint=databricks_resources.get("embedding_endpoint_name")
# MAGIC )
# MAGIC
# MAGIC # Get deployment client
# MAGIC deploy_client = get_deploy_client("databricks")
# MAGIC
# MAGIC # Get parent configuration
# MAGIC parent_splits_table_schema = retriever_config.get("parent_splits_table_schema")
# MAGIC
# MAGIC # Create custom LangChain retriever
# MAGIC retriever = DatabricksParentDocumentRetriever(
# MAGIC     vs_index=vs_index,
# MAGIC     embedding_model=embedding_model,
# MAGIC     deploy_client=deploy_client,
# MAGIC     feature_endpoint_name=databricks_resources.get("feature_serving_endpoint_name"),
# MAGIC     parent_id_key=parent_splits_table_schema.get("primary_key"),
# MAGIC     content_col=parent_splits_table_schema.get("text_col"),
# MAGIC     filter_col=parent_splits_table_schema.get("document_name_col"),
# MAGIC     source_col=parent_splits_table_schema.get("document_uri")
# MAGIC )
# MAGIC
# MAGIC # Required to:
# MAGIC # 1. Enable the Agent Framework Review App to properly display retrieved chunks
# MAGIC # 2. Enable evaluation suite to measure the retriever
# MAGIC mlflow.models.set_retriever_schema(
# MAGIC     primary_key=parent_splits_table_schema.get("primary_key"),
# MAGIC     text_column=parent_splits_table_schema.get("text_col"),
# MAGIC     doc_uri=parent_splits_table_schema.get("document_uri"),
# MAGIC     other_columns=[
# MAGIC         parent_splits_table_schema.get("document_name_col"),
# MAGIC         parent_splits_table_schema.get("primary_key"),
# MAGIC     ],
# MAGIC )
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt
# MAGIC def format_context(docs):
# MAGIC     chunk_template = retriever_config.get("chunk_template")
# MAGIC     chunk_contents = [
# MAGIC         chunk_template.format(
# MAGIC             chunk_text=d.page_content,
# MAGIC         )
# MAGIC         for d in docs
# MAGIC     ]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC
# MAGIC # Prompt Template for generation
# MAGIC prompt = ChatPromptTemplate.from_messages(
# MAGIC     [
# MAGIC         ("system", llm_config.get("llm_prompt_template")),
# MAGIC         # Note: This chain does not compress the history, so very long converastions can overflow the context window.
# MAGIC         MessagesPlaceholder(variable_name="formatted_chat_history"),
# MAGIC         # User's most current question
# MAGIC         ("user", "{question}"),
# MAGIC     ]
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # Format the converastion history to fit into the prompt template above.
# MAGIC def format_chat_history_for_prompt(chat_messages_array):
# MAGIC     history = extract_chat_history(chat_messages_array)
# MAGIC     formatted_chat_history = []
# MAGIC     if len(history) > 0:
# MAGIC         for chat_message in history:
# MAGIC             if chat_message["role"] == "user":
# MAGIC                 formatted_chat_history.append(HumanMessage(content=chat_message["content"]))
# MAGIC             elif chat_message["role"] == "assistant":
# MAGIC                 formatted_chat_history.append(AIMessage(content=chat_message["content"]))
# MAGIC     return formatted_chat_history
# MAGIC
# MAGIC # Prompt Template for query rewriting to allow converastion history to work - this will translate a query such as "how does it work?" after a question such as "what is spark?" to "how does spark work?".
# MAGIC query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.
# MAGIC
# MAGIC Chat history: {chat_history}
# MAGIC
# MAGIC Question: {question}"""
# MAGIC
# MAGIC query_rewrite_prompt = PromptTemplate(
# MAGIC     template=query_rewrite_template,
# MAGIC     input_variables=["chat_history", "question"],
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # FM for generation
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC     extra_params=llm_config.get("llm_parameters"),
# MAGIC )
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
# MAGIC         "formatted_chat_history": itemgetter("messages") | RunnableLambda(format_chat_history_for_prompt),
# MAGIC         "filter": itemgetter("messages") | RunnableLambda(extract_filter_value),
# MAGIC     }
# MAGIC     | RunnablePassthrough()
# MAGIC     | {
# MAGIC         "context": RunnableBranch(  # Only re-write the question if there is a chat history
# MAGIC             (
# MAGIC                 lambda x: len(x["chat_history"]) > 0, 
# MAGIC                 query_rewrite_prompt | model | StrOutputParser(),
# MAGIC             ),
# MAGIC             RunnableLambda(lambda x: {"content": x["question"]} if x["filter"] is None else {"content": x["question"], "filter": x["filter"]})
# MAGIC         )
# MAGIC         | retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC         "formatted_chat_history": itemgetter("formatted_chat_history"),
# MAGIC         "question": itemgetter("question"),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC # Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log model 
# MAGIC
# MAGIC The following code block will log the langchain and the dependencies. The default for ```lc_model```parameter is ```conversational_chain.py```. If you want to use a simple Q&A chain you can also use ```chain.py```. But be aware that the front-end app is currently build for conversational chains only. 

# COMMAND ----------


# Set the registry URI to Unity Catalog 
mlflow.set_registry_uri("databricks-uc")

# Create the model registry in Unity Catalog
registered_model_name = f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.conv_chain_model"

# The agent framework can't handle an additional filter criteria at the current stage. That's why we log the model input example without the filter. Still, filters are supported by the model serving endpoint.
input_example = {
        "messages": [{"content": "What are the conditions of my travel cancellation insurance?", "role": "user"}]
    }

# Log the model to MLflow
with mlflow.start_run(run_name=f"large_doc_rag_conversational") as l:
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'src/conversational_chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config='rag_chain_config.yaml',  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        code_paths=["src"],
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        registered_model_name=registered_model_name
    )

# COMMAND ----------

# DBTITLE 1,Invoke chain with filter
# Test the chain locally
version = logged_chain_info.registered_model_version
chain = mlflow.langchain.load_model(f"models:/{registered_model_name}/{version}")

# Test chain with filter criteria
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

# DBTITLE 1,Invoke chain without filter
# Test chain without filter criteria
chain.invoke(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Chain Endpoint + Review App

# COMMAND ----------

from databricks import agents

instructions_to_reviewer = f"""

### Instructions for Testing our Copilot Assistant

Your expertise and feedback are crucial for refining our Insurance Copilot. By providing detailed insights and corrections, you help us enhance the accuracy and relevance of the information provided to our insurance clients. Your input is vital in identifying areas for improvement and ensuring the Copilot meets the specific needs of the insurance industry.

1. **Diverse Insurance Scenarios**:
   - Please test the Copilot with a wide range of insurance-related queries. Include questions about policy types, coverage details, claims processes, and regulatory compliance that our clients typically ask. This helps us ensure the Copilot can effectively handle real-world insurance inquiries.

2. **Evaluation of Responses**:
   - After each query, use the provided feedback tools to assess the Copilot's response.
   - If you find any inaccuracies or areas for improvement in the answers, please use the "Edit Answer" feature to provide corrections. Your industry-specific knowledge will help us fine-tune the Copilot's accuracy and relevance.

3. **Assessment of Referenced Insurance Documents**:
   - Carefully review the insurance documents, policies, or clauses that the Copilot references in its responses.
   - Utilize the thumbs up/down feature to indicate whether the referenced material is pertinent to the insurance query. A thumbs up indicates relevance and accuracy, while a thumbs down suggests the referenced material was not applicable or potentially misleading.

4. **Compliance and Up-to-Date Information**:
   - Pay special attention to responses involving insurance regulations, policy terms, or industry standards. Ensure the information provided is current and compliant with the latest insurance laws and practices.

5. **Clarity for Non-Experts**:
   - Evaluate whether the Copilot's explanations of complex insurance terms or processes are clear enough for clients who may not have in-depth insurance knowledge.

Your dedication to testing this Insurance Copilot is invaluable. Your insights will help us deliver a robust, accurate, and user-friendly tool that meets the specific needs of insurance professionals and clients alike. Thank you for your time and expertise in this critical development phase."""


# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(
    model_name=registered_model_name,
    model_version=version,
    scale_to_zero=True,
    environment_vars={
        "DATABRICKS_TOKEN": f"{{{{secrets/{secrets_config.get('secret_scope')}/{secrets_config.get('secret_key')}}}}}",
        "DATABRICKS_HOST": f"{databricks_resources.get('host')}"
    },
)

# Add the user-facing instructions to the Review App
agents.set_review_instructions(registered_model_name, instructions_to_reviewer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant stakeholders access to the Mosaic AI Agent Evaluation App
# MAGIC
# MAGIC Now, grant your stakeholders permissions to use the Review App. To simplify access, stakeholders do not require to have Databricks accounts.

# COMMAND ----------

# Create the model registry in Unity Catalog
registered_model_name = f"{databricks_resources.get('catalog')}.{databricks_resources.get('schema')}.conv_chain_model"

# COMMAND ----------

from databricks import agents

user_list = ['<email adress>'] # Add user's that will have access to the review App. This requires that user e-mail is part of the SSO Directory. 

# Set the permissions.
agents.set_permissions(model_name=registered_model_name, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------


