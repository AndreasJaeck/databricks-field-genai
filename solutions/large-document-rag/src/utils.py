
import os
import requests
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from urllib.parse import quote


class DatabricksParentDocumentRetriever(BaseRetriever):
    """
    A class for retrieving relevant parent documents based on a query using Databricks Vector Search and Online Table Serving.

    Args:
        vs_index (DatabricksVectorSearch): The Databricks Vector Search index object.
        embedding_model (DatabricksEmbeddings): The Databricks Embeddings model for generating query embeddings.
        deploy_client (object): The deployment client for querying the feature store.
        feature_endpoint_name (str): The name of the feature serving endpoint.
        parent_id_key (str): The key for the parent document ID in the feature store.
        content_col (str): The column name for the parent document content in the feature store.
        filter_col (str): The column name for the filter value in the feature store.
        source_col (str): The column name for the source information in the feature store.

    Attributes:
        vs_index (DatabricksVectorSearch): The Databricks Vector Search index object.
        embedding_model (DatabricksEmbeddings): The Databricks Embeddings model for generating query embeddings.
        deploy_client (object): The deployment client for querying the feature store.
        feature_endpoint_name (str): The name of the feature serving endpoint.
        parent_id_key (str): The key for the parent document ID in the feature store.
        content_col (str): The column name for the parent document content in the feature store.
        filter_col (str): The column name for the filter value in the feature store.
        source_col (str): The column name for the source information in the feature store.
    """

    vs_index: object
    embedding_model: object
    deploy_client: object
    feature_endpoint_name: str
    parent_id_key: str
    content_col: str
    filter_col: str
    source_col: str

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieves relevant parent documents based on a query.

        Args:
            query (str or dict): The query string or a dictionary containing the question and optional filter value.
            run_manager (CallbackManagerForRetrieverRun): The run manager for managing callbacks during retrieval.

        Returns:
            List[Document]: A list of relevant parent documents.
        """
        
        # Generate embedding vector from query content
        if "content" in query:
            embedding_vector = self.embedding_model.embed_query(query["content"])
        else:
            embedding_vector = self.embedding_model.embed_query(query)

        # Set filter
        if "filter" in query: 
            filters = {self.filter_col: query["filter"]}
        else:
            filters = None

        # Perform similarity search in the vector index to find matching parent document IDs
        resp = self.vs_index.similarity_search(
            columns=[self.parent_id_key],
            query_vector=embedding_vector,
            num_results=3,
            filters=filters,
        )

        data = resp.get("result", {}).get("data_array", None)

        # Handle case where no matching documents are found
        if not data:
            result_docs = [Document("no context found")]
        else:
            # Create unique set of IDs so we are not retrieving the same document twice
            parent_document_ids = list(
                set([int(document_id) for document_id, distance in data])
            )

            ## Get parent documents with parent IDs
            # Put IDs into TF format to query the feature serving endpoint
            ids = {
                "dataframe_records": [
                    {self.parent_id_key: id} for id in parent_document_ids
                ]
            }

            # Query the feature serving endpoint to retrieve parent document content
            parent_content = self.deploy_client.predict(endpoint=self.feature_endpoint_name, inputs=ids)
    
            # Convert retrieved content into Document objects
            result_docs = [
                Document(
                    page_content=doc[self.content_col],
                    metadata={
                        self.source_col: doc[self.source_col],
                        self.filter_col: doc[self.filter_col],
                        self.parent_id_key: doc[self.parent_id_key],
                    },
                )
                for doc in parent_content["outputs"]
            ]

        return result_docs
