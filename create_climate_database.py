import os
from dotenv import load_dotenv
import re

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient 
from azure.ai.projects.models import DatasetVersion


endpoint = "https://nadine-ai-resource.services.ai.azure.com/api/projects/firstProject"
connection_name = "nadineaihub4723196171"  # Name of Azure Storage connection

dataset_name = "climateDatabase"


script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(script_dir, "datasets")
weather_file = os.path.join(data_folder, "adaptations.txt")

with DefaultAzureCredential() as credential:
    with AIProjectClient(
        endpoint=endpoint,
        credential=credential,
    ) as client:
        dataset = client.datasets.upload_file(
        name = dataset_name,
        file_path=weather_file,
        version="3.0",
        connection_name=connection_name
        )
    





