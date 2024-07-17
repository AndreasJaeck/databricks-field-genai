# Copilot RAG Solution Accelerator for Large Documents

## Table of Contents
* [Why a Copilot for Large Documents?](#why-a-copilot-for-large-documents)
* [About this Solution Accelerator](#about-this-solution-accelerator)
* [Vector Databases vs Large Context Windows](#vector-databases-vs-large-context-windows)
* [Structure](#structure)
* [Architecture](#architecture)
* [Disclaimer](#disclaimer)
* [Outlook](#outlook)

## Why a Copilot for Large Documents?
Many industries possess extensive knowledge bases but struggle to leverage that knowledge systematically. LLMs with access to this knowledge have proven to be excellent solutions to this business challenge. However, many current low-code or no-code solutions struggle to provide comprehensive answers when documents are large and complex. We propose a solution that combines large context window retrieval with vector search to yield superior results in verticals such as:

1. Finance (contracts)
2. Manufacturing (technical manuals)
3. Public Service (legal papers)

### Vector Databases vs Large Context Windows

![Vector Databases vs Large Context Windows](img/hybrid-rag.webp)

Our hybrid approach leverages the strengths of both techniques, providing a robust and flexible solution for handling large documents and specific queries that need to find the proverbial needle in the haystack.

### Architecture

![RAG Solution Architecture](img/vector-db-vs-large-context.webp)

Our proposed solution architecture combines the power of vector databases and large context windows to create a robust and efficient RAG system:

1. Document Ingestion: Large documents are processed and split into manageable parent chunks.
2. Vector Embedding: Parent chunks are further broken down into a chunk size well-suited for embeddings and vector databases.
3. Query Processing: User queries are embedded, and the vector database searches for semantically similar content.
4. Context Assembly: The vector database returns the ID of the parent document from which the semantically similar content was derived.
5. Prompt Creation: The parent document is retrieved from the database, and the final prompt is generated.
6. Response Generation: The LLM generates a response based on the provided context and user query.
7. Result Presentation: The generated response is presented to the user through the front-end interface.

This architecture allows for efficient retrieval of relevant information from large document collections while maintaining the ability to process and understand complex relationships within the retrieved context.

## About this Solution Accelerator
This project contains code to deploy a RAG (Retrieval Augmented Generation) solution on Databricks. It combines the advantages of Vector Search with the large window capability of recent LLMs (Large Language Models). This solution is optimal for scenarios requiring large context windows to answer open questions while maintaining the ability to find specific information within vast datasets.

### Structure
The following notebooks will create a Hybrid RAG Chatbot with a Review UI and a Front-End for end-user interaction:

* [01-Config](./01-Config.py) - Create the configuration for the project. **Please adjust parameters here!**
* [02-Create-Tables](./02-Create-Tables.py) - Creates the necessary tables.
* [03-Create-Document-Job](./03-Create-Document-Job.py) - Creates a job that can be scheduled or triggered to update tables and index.
* [03b-Load-Documents](./03b-Load-Documents.py) - Notebook that runs as a job to update tables and index.
* [04-Create-Endpoints](./04-Create-Endpoints.py) - Creates RAG-Chain, Endpoints, and Review UI.
* [05-Deploy-Frontend-Lakehouse-App](./05-Deploy-Frontend-Lakehouse-App.py) - Creates a Gradio Front-End App and deploys it as a Lakehouse App (Private Preview)

### How to use:
Watch the following videos for a complete guide to the rollout:

#### [01-Config](./01-Config.py)
#### [02-Create-Tables](./02-Create-Tables.py)
#### [03-Create-Document-Job](./03-Create-Document-Job.py) 
#### [03b-Load-Documents](./03b-Load-Documents.py)
#### [04-Create-Endpoints](./04-Create-Endpoints.py)
#### [05-Deploy-Frontend-Lakehouse-App](./05-Deploy-Frontend-Lakehouse-App.py)

## Disclaimer 
This solution is not intended for B2C use cases or in scenarios where the Copilot response is used without supervision.

## Outlook
We plan to add the following features in the near future:

* Judge Evaluation Workflows
* Monitoring Dashboard
* Multi-Modal Document Support
* Deployment as Databricks Asset Bundles (DAB)
