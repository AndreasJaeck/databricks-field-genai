# Large Document Retrieval Augumented Generation

## Table on contents
* [Intro](#intro)


## Intro
This project comes with example code to deploy a RAG solution on Databricks that combines the advantages of Vector Search with large window capability of LLM's. This solution will work best in scenarios where large context windows are required to answer comprehensive questions (e.g legal documents, technical manuals)

## Structure

* [01-Config](./01-Config.py) Creates the configuration for the project
* [02-Create-Tables](./02-Create-Tables.py) Creates the Delta Tables
* [03-Create-Document-Job](./3-Create-Document-Job.py) Creates a job that will load documents from the Volume, processes and syncs to the online table and vector index.
* [04-Build-Parent-Document-Retirever](./04-Build-Parent-Document-Retirever.py) 




