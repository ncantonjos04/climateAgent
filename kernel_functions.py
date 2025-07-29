import os 
import asyncio
import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datetime import datetime, timedelta


from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import  kernel_function

# Load .env file 
load_dotenv()

# Connect to the NASA POWER API to get accurate weather data in the chosen location
# returns Total Precipitation (T2M) and Temperature at 2 Meters (T2M)

@kernel_function
async def get_NASA_data (location: str, start_year: int, end_year: int):
    api_key= os.getenv("GEO_API_KEY")
    url = f"https://api.opencagedata.com/geocode/v1/json"
    params = {"q": location, "key":api_key}
    response = requests.get(url, params = params)
    data = response.json()
    coords = data["results"][0]["geometry"]
    
    def blocking_fetch():
    # Connect to NASA POWER API with url using the above parameters
        base_url = (
            f"https://power.larc.nasa.gov/api/temporal/monthly/point?"
            f"start={start_year}&end={end_year}"
            f"&latitude={coords["lat"]}&longitude={coords["lng"]}"
            f"&community=ag"
            f"&parameters=T2M,PRECTOT"
            f"&format=csv&header=false"
        )

    # Write results to a .txt file  (including the header)


        data = response.text
        with open('./datasets/weather_data.txt', "a") as file:
            file.write(f"{data}\n")
        return data
    data = await asyncio.get_event_loop().run_in_executor(None, blocking_fetch)
    return data
        
@kernel_function
async def get_forecast(location: str, forecast_date):  # date: YYYY-MM-DD
    api_key=os.getenv("GEO_API_KEY")
    url = f"https://api.opencagedata.com/geocode/v1/json"
    params = {"q": location, "key":api_key}
    response = requests.get(url, params = params)
    data = response.json()
    coords = data["results"][0]["geometry"]
    lat = coords["lat"]
    lon = coords["lng"]

    now = datetime.now()
    if isinstance(forecast_date, datetime):
        forecast_date = forecast_date.strftime("%Y-%m-%d")
    elif isinstance(forecast_date, bool): # 1 or 0
        forecast_date = int(forecast_date)
    elif (isinstance(forecast_date, str) and "-" not in forecast_date): # "1" or "0"
        forecast_date = int(forecast_date)
    elif isinstance(forecast_date,int):
        if forecast_date == 0:
            forecast_date = now
        elif forecast_date != 0:
            forecast_date = datetime.today() + timedelta(days=forecast_date)
        forecast_date = forecast_date.strftime("%Y-%m-%d")

    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "appid": api_key,
        "lat": lat,
        "lon": lon,
        "exclude": "minutely,hourly,alerts",
        "units": "metric"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"

    data = response.json()

    # Find the matching forecast day
    forecast_day = None
    for day in data.get("daily", []):
        dt = datetime.fromtimestamp(day["dt"]).strftime("%Y-%m-%d")
        if dt == forecast_date:
            forecast_day = day
            break

    if not forecast_day:
        return f"No forecast found for {forecast_date}."

    summary = {
        "date": forecast_date,
        "temp_day": forecast_day["temp"]["day"],
        "temp_night": forecast_day["temp"]["night"],
        "weather": forecast_day["weather"][0]["description"],
        "precipitation_mm": forecast_day.get("rain", 0.0)
    }

    return summary

@kernel_function
async def get_adaptations():
    with open('./datasets/adaptations.txt', "r") as file:
        content = file.read()
        return content

