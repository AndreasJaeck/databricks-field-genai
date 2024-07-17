
import os
from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.deployments import get_deploy_client
from operator import itemgetter
from databricks.vector_search.client import VectorSearchClient
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import DatabricksEmbeddings
from src.utils import DatabricksParentDocumentRetriever


# Workspace client will automatically pick up credentials from the environment var 'DATABRICKS_HOST' and 'DATABRICKS_TOKEN'
w = WorkspaceClient()

## Enable MLflow Tracing
mlflow.langchain.autolog()

#Get the configuration from the local config file
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

databricks_resources = model_config.get("databricks_resources")
secrets_config = model_config.get("secrets_config")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")


# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# Return the string filter value of the most recent message from the user
def extract_filter_value(chat_messages_array):
    if "filter" in chat_messages_array[-1]:
        return chat_messages_array[-1]["filter"]
    else:
        return None
    
    
# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]
      
## Get Vector Index
vsc = VectorSearchClient(
    disable_notice=True, 
    workspace_url=os.environ["DATABRICKS_HOST"],
    personal_access_token=os.environ["DATABRICKS_TOKEN"],
    )

vs_index = vsc.get_index(
    endpoint_name=databricks_resources.get("vector_search_endpoint_name"),
    index_name=databricks_resources.get("vector_search_index"),
)

## Get embedding endpoint
embedding_model = DatabricksEmbeddings(
    endpoint=databricks_resources.get("embedding_endpoint_name")
)

# Get deployment client
deploy_client = get_deploy_client("databricks")

# Get parent configuration
parent_splits_table_schema = retriever_config.get("parent_splits_table_schema")

# Create custom LangChain retriever
retriever = DatabricksParentDocumentRetriever(
    vs_index=vs_index,
    embedding_model=embedding_model,
    deploy_client=deploy_client,
    feature_endpoint_name=databricks_resources.get("feature_serving_endpoint_name"),
    parent_id_key=parent_splits_table_schema.get("primary_key"),
    content_col=parent_splits_table_schema.get("text_col"),
    filter_col=parent_splits_table_schema.get("document_name_col"),
    source_col=parent_splits_table_schema.get("document_uri")
)

# Required to:
# 1. Enable the Agent Framework Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
mlflow.models.set_retriever_schema(
    primary_key=parent_splits_table_schema.get("primary_key"),
    text_column=parent_splits_table_schema.get("text_col"),
    doc_uri=parent_splits_table_schema.get("document_uri"),
    other_columns=[
        parent_splits_table_schema.get("document_name_col"),
        parent_splits_table_schema.get("primary_key"),
    ],
)

# Method to format the docs returned by the retriever into the prompt
def format_context(docs):
    chunk_template = retriever_config.get("chunk_template")
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
        )
        for d in docs
    ]
    return "".join(chunk_contents)


# Prompt Template for generation
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", llm_config.get("llm_prompt_template")),
        # Note: This chain does not compress the history, so very long converastions can overflow the context window.
        MessagesPlaceholder(variable_name="formatted_chat_history"),
        # User's most current question
        ("user", "{question}"),
    ]
)


# Format the converastion history to fit into the prompt template above.
def format_chat_history_for_prompt(chat_messages_array):
    history = extract_chat_history(chat_messages_array)
    formatted_chat_history = []
    if len(history) > 0:
        for chat_message in history:
            if chat_message["role"] == "user":
                formatted_chat_history.append(HumanMessage(content=chat_message["content"]))
            elif chat_message["role"] == "assistant":
                formatted_chat_history.append(AIMessage(content=chat_message["content"]))
    return formatted_chat_history

# Prompt Template for query rewriting to allow converastion history to work - this will translate a query such as "how does it work?" after a question such as "what is spark?" to "how does spark work?".
query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}"""

query_rewrite_prompt = PromptTemplate(
    template=query_rewrite_template,
    input_variables=["chat_history", "question"],
)


# FM for generation
model = ChatDatabricks(
    endpoint=databricks_resources.get("llm_endpoint_name"),
    extra_params=llm_config.get("llm_parameters"),
)

# RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        "formatted_chat_history": itemgetter("messages") | RunnableLambda(format_chat_history_for_prompt),
        "filter": itemgetter("messages") | RunnableLambda(extract_filter_value),
    }
    | RunnablePassthrough()
    | {
        "context": RunnableBranch(  # Only re-write the question if there is a chat history
            (
                lambda x: len(x["chat_history"]) > 0, 
                query_rewrite_prompt | model | StrOutputParser(),
            ),
            RunnableLambda(lambda x: {"content": x["question"]} if x["filter"] is None else {"content": x["question"], "filter": x["filter"]})
        )
        | retriever
        | RunnableLambda(format_context),
        "formatted_chat_history": itemgetter("formatted_chat_history"),
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)
