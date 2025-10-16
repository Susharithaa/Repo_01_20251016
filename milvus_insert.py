from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
import os
import pandas as pd
import json
from langchain_openai import AzureOpenAIEmbeddings
import openai
from langchain_community.vectorstores import Milvus
from langchain_openai import AzureChatOpenAI
import re
import time


Embeddings = {
    "openai_api_type": "azure",
    "model": "nextgen-embedding-model",
    "deployment": "nextgen-embedding-model",
    "openai_api_base": "https://nextgen-azure-openai.openai.azure.com/",
    "openai_api_version": "2024-02-01",
    "openai_api_key": "a3efe3e1fe1f4b6ca6f9d6d1a23f8991"
}
embedding_function = AzureOpenAIEmbeddings(
    openai_api_type=Embeddings['openai_api_type'],
    model=Embeddings['model'],
    deployment=Embeddings['deployment'],
    azure_endpoint=Embeddings['openai_api_base'],
    openai_api_version=Embeddings['openai_api_version'],
    openai_api_key=Embeddings['openai_api_key'])
chat_model = {
    "openai_api_type": "azure",
    "deployment_name": 'nextgen-dev-gpt-4o',
    "model_name": "gpt-4o",
    "openai_api_base": "https://tools-openai.openai.azure.com/",
    "openai_api_version": "2024-05-01-preview",
    "openai_api_key": "0e7723c48eda490e975c0c2bb6a13ce5",
    "temperature": 0
}
llm = AzureChatOpenAI(openai_api_type=chat_model['openai_api_type'],
                      deployment_name=chat_model['deployment_name'],
                      model_name=chat_model['model_name'],
                      azure_endpoint=chat_model['openai_api_base'],
                      openai_api_version=chat_model['openai_api_version'],
                      openai_api_key=chat_model['openai_api_key'],
                      temperature=0)
milvus_credentials = {"host": "10.162.101.41", "port": "19530", "db_name": "data_dialogue",
                      "collection": "business_context", }  ###### change here

index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "index_name" : "vector_index",
    "params" : { "M": 8, "efConstruction" :64 }
}

milvus_db = Milvus(embedding_function,
                   connection_args={"host": milvus_credentials['host'], "port": milvus_credentials['port'],
                                    "db_name": milvus_credentials['db_name']},
                   collection_name=milvus_credentials['collection'], 
                   consistency_level = 'Strong', 
                   text_field = 'values', 
                   index_params = index_params, 
                   auto_id=True)

chunk_size, chunk_overlap = 2000, 200


def generate_summary(row):
    prompt = f"""Summarize key details from the table and column descriptions to quickly identify the most relevant ones for specific questions.
    **Input Field**
    Table Info: {row['Table Description']} 
    Column Description : {row['Column Description']} """
    # Replace with your LLM API call
    prompt_summary = llm.invoke(prompt)

    summary = prompt_summary.content
    return summary


def load_files(file_path):
    file_data = pd.read_excel(file_path)
    file_data['entities'] = file_data['Entities'].astype(str)
    file_data['description'] = file_data['Description'].astype(str)
    file_data['values'] = file_data['Values'].astype(str)
    data = file_data[['entities', 'description','values']]
    # data.to_excel("Summarised_Details.xlsx")
    loader = DataFrameLoader(data, page_content_column="values")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    page_splits = text_splitter.split_documents(pages)
    print(page_splits[0])
    print('Total rows after chunking: ', len(page_splits))
    batch_size = 10
    page_split_batches = [page_splits[i:i + batch_size] for i in range(0, len(page_splits), batch_size)]
    for index, batch in enumerate(page_split_batches):
        print(batch)
        milvus_db.add_documents(batch)
        print(f"Batch {index + 1}: {len(batch)}")
        time.sleep(10)
    print('Data inserted Successfully!')


file_path = r"./org_context.xlsx"
load_files(file_path)

