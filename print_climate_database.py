import os
from dotenv import load_dotenv
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.storage.blob import ContainerClient, BlobClient

load_dotenv()

endpoint = "https://nadine-ai-resource.services.ai.azure.com/api/projects/firstProject"
connection_name = "nadineaihub4723196171"  # Name of Azure Storage connection
dataset_name = "climateDatabase"
dataset_version = "5.0"

def is_blob_url(url):
    """
    Heuristically decide if the URL points to a blob (file)
    or just a container/folder.
    """
    # URL path after container name typically blob_path/blobname
    # If path has more than 1 segment after container, treat as blob URL.
    parsed = urlparse(url)
    path_parts = parsed.path.lstrip('/').split('/')
    # The first segment is container name
    return len(path_parts) > 1

def list_all_files_in_container(container_url, credential):
    parsed = urlparse(container_url)
    container_name = parsed.path.lstrip('/').split('/')[0]
    account_url = f"https://{parsed.netloc}"
    container_client = ContainerClient(account_url, container_name, credential)
    
    files = []
    blobs = container_client.list_blobs()
    for blob in blobs:
        if not blob.name.endswith('/'):
            files.append(blob.name)
    return files, container_client

def print_file_contents(container_client, blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    stream = blob_client.download_blob()
    content = stream.readall().decode('utf-8')
    
    print(f"--- {blob_name} ---")
    print(content)
    print()  # blank line after file content

def print_single_blob(blob_url, credential):
    parsed = urlparse(blob_url)
    container_name = parsed.path.lstrip('/').split('/')[0]
    blob_path = '/'.join(parsed.path.lstrip('/').split('/')[1:])
    account_url = f"https://{parsed.netloc}"
    blob_client = BlobClient(account_url, container_name, blob_path, credential=credential)
    stream = blob_client.download_blob()
    content = stream.readall().decode('utf-8')

    print(f"--- {blob_path} ---")
    print(content)
    print()

with DefaultAzureCredential() as credential:
    with AIProjectClient(endpoint, credential) as client:
        dataset = client.datasets.get(name=dataset_name, version=dataset_version)
        data_uri = dataset._data.get("dataUri")
        if not data_uri:
            raise RuntimeError("Could not find dataset URI in _data['dataUri']")
        
        print(f"Dataset Storage URL: {data_uri}")

        if is_blob_url(data_uri):
            # dataUri points directly to a blob file
            print("Dataset points to a single file (blob). Printing its contents:")
            print_single_blob(data_uri, credential)
        else:
            # dataUri points to container or folder, list all blobs
            file_list, container_client = list_all_files_in_container(data_uri, credential)
            print("Files in dataset:")
            for file in file_list:
                print(f"- {file}")
            print()
            for file in file_list:
                print_file_contents(container_client, file)
