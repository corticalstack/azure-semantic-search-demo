# 🔍 Azure Semantic Search for COVID-19 Research

A comprehensive semantic search solution for COVID-19 research papers using Azure Machine Learning and neural language models.

## Description

This repository contains an end-to-end pipeline for processing, analyzing, and enabling semantic search over the COVID-19 Open Research Dataset (CORD-19). It leverages Azure Machine Learning for ETL processes, neural language models for semantic understanding, and Elasticsearch for efficient retrieval.

The system processes scientific papers, extracts meaningful information, generates embeddings using transformer models, and indexes them for semantic search capabilities. This allows researchers to find relevant COVID-19 research papers using natural language queries.

## Prerequisites

- Azure Machine Learning workspace
- Azure Storage Account
- Azure Key Vault (for storing credentials)
- PostgreSQL database
- Elasticsearch instance
- Python 3.6+
- CUDA-compatible GPU for model training

## Features

- **ETL Pipeline**: Processes CORD-19 dataset, extracts text, detects language, and performs data cleansing
- **Text Analysis**: Performs stopword removal, lemmatization, and topic modeling on research papers
- **Author & Article Metrics**: Calculates metrics for authors and articles, including citation counts and PageRank
- **Neural Search**: Uses transformer models (ANCE-MSMarco) to generate semantic embeddings
- **Query Generation**: Automatically generates training queries from paragraphs using T5 models
- **Elasticsearch Integration**: Indexes embeddings and metadata for efficient semantic search

## Architecture

The system is organized into two main components:

1. **ETL Pipeline** (`azureml/etl/cord19/`): Processes raw CORD-19 data
   - Extracts text from JSON files
   - Detects language
   - Creates hash indices
   - Removes stopwords and performs lemmatization
   - Identifies clinical trials
   - Performs feature engineering
   - Calculates author and article metrics
   - Conducts topic modeling
   - Updates database with processed data

2. **Model Pipeline** (`azureml/model/cord19/`): Trains and deploys semantic search models
   - Trains neural models on MS MARCO dataset
   - Generates synthetic queries for training
   - Embeds corpus paragraphs using trained models
   - Indexes embeddings in Elasticsearch
   - Provides search functionality

## Setup Guide

1. **Configure Azure ML Workspace**:
   - Create an Azure ML workspace
   - Set up compute clusters (CPU for ETL, GPU for model training)
   - Configure datastores for input/output data

2. **Environment Setup**:
   - Use the provided Docker image (`azureml-env-base-sd:latest`) or create a custom environment
   - Required packages: azureml, pandas, numpy, torch, sentence-transformers, elasticsearch, etc.

3. **Database Configuration**:
   - Set up PostgreSQL database with required tables
   - Configure connection strings in Key Vault

4. **Elasticsearch Setup**:
   - Deploy Elasticsearch instance
   - Configure connection details in Key Vault

5. **Pipeline Execution**:
   - Run ETL pipeline: `azureml_pipeline_cord19_transform.ipynb`
   - Train model: `azureml_pipeline_cord19_model_train_ance_msmarco_passage.ipynb`
   - Index corpus: `azureml_pipeline_cord19_embed_to_es.ipynb`
   - Generate queries: `azureml_pipeline_cord19_model_generate_queries.ipynb`

## Resources

- [CORD-19 Dataset](https://www.semanticscholar.org/cord19)
- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Sentence Transformers](https://www.sbert.net/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)

## FAQ

### How does the semantic search work?

The system uses neural language models to convert both queries and document paragraphs into dense vector embeddings. When a search is performed, the system finds documents whose embeddings are most similar to the query embedding, allowing for semantic matching beyond simple keyword matching.

### What models are used for embeddings?

The system primarily uses the ANCE-MSMarco passage model (`castorini/ance-msmarco-passage`), which is fine-tuned on the MS MARCO dataset for passage retrieval tasks.

### How is the CORD-19 data processed?

The ETL pipeline extracts text from JSON files, performs language detection, removes stopwords, applies lemmatization, and conducts topic modeling. It also calculates various metrics like citation counts and identifies clinical trials.

### Can I use this for other datasets?

While the system is designed for the CORD-19 dataset, it can be adapted for other scientific literature datasets with modifications to the ETL pipeline.
