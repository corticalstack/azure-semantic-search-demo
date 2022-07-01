FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04

ARG AZUREML_SDK_VERSION=1.43.0
ARG INFERENCE_SCHEMA_VERSION=1.4.1

ARG az_sd_psql_host
ARG az_sd_psql_user
ARG az_sd_psql_pwd
ARG az_sd_es_host
ARG az_sd_es_user
ARG az_sd_es_pwd

ENV AZ_SD_PSQL_HOST=$az_sd_psql_host
ENV AZ_SD_PSQL_USER=$az_sd_psql_user
ENV AZ_SD_PSQL_PWD=$az_sd_psql_pwd
ENV AZ_SD_ES_HOST=$az_sd_es_host
ENV AZ_SD_ES_USER=$az_sd_es_user
ENV AZ_SD_ES_PWD=$az_sd_es_pwd

RUN conda update --all --yes

RUN apt-get update --fix-missing

RUN conda install pytorch=1.6 torchvision cudatoolkit=10.1 -c pytorch

RUN pip install azureml-core &&\
    pip install azureml-pipeline

RUN pip install azure-storage-blob

RUN pip install scipy==1.8.1 &&\
    pip install scispacy==0.5.0 &&\
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_lg-0.5.0.tar.gz &&\
    pip install spacy-langdetect &&\
    pip install nltk &&\
    pip install gensim

RUN pip install matplotlib --ignore-installed &&\
    pip install plotly &&\
    pip install pyLDAvis &&\
    pip install ruamel-yaml &&\
    pip install networkx &&\
    pip install seaborn &&\
    pip install plotly

RUN apt-get install libpq-dev -y &&\
    pip install psycopg2 &&\
    pip install sqlalchemy

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install python3-tk -y &&\
    pip install opencv-python &&\
    pip install camelot-py[cv] &&\
    conda install -c conda-forge ghostscript -y &&\
    conda install -c conda-forge faiss

RUN pip install xlrd &&\
    pip install elasticsearch &&\
    pip install "modin[ray]" &&\
    pip install sentencepiece &&\
    conda install -c huggingface tokenizers=0.9.4 transformers=4.2.2 &&\
    pip install sentence-transformers