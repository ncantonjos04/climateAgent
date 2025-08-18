# AgroAskAI

AgroAskAI is an AI-powered assistant designed to answer **agriculture-related questions**. It provides insights on farming, weather forecasts, and historical climate data to support decision-making in agriculture.

# Setup
## Requirements
   - Python 3.11+
   - A GitHub Account - For Access to the GitHub Models Marketplace
## Install dependencies
  We have included a ```requirements.txt``` file in the root of this repository that contains all the required Python packages to run the code samples.
    
  You can install them by running the following command in your terminal at the root file:
  
  ```pip install -r requirements.txt```
## **Create your .env file**

  Run the following command in your terminal at the root file:
   ```cp .env.example .env```

  This will copy the example file and create a .env in your directory and where you fill in the values for the environment variables.

  With your tokens copied, open the .env file in a text editor and paste the required tokens into their fields.

## Retrieve a GitHub Personal Access Token ```OPENAI_API_KEY```

This project leverages the GitHub Models Marketplace, providing free access to Large Language Models (LLMs) that you will use to build AI Agents.

To use the GitHub Models, you will need to create a [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

This can be done by going to your <a href="https://github.com/settings/personal-access-tokens" target="_blank">Personal Access Tokens settings</a> in your GitHub Account.

Please follow the [Principle of Least Privilege](https://docs.github.com/en/get-started/learning-to-code/storing-your-secrets-safely) when creating your token. This means you should only give the token the permissions it needs to run the code samples in this course.

1. Select the `Fine-grained tokens` option on the left side of your screen.

    Then select `Generate new token`.


2. Enter a descriptive name for your token.


3. Limit the token's scope to your fork of this repository.


4. Restrict the token's permissions: Under **Permissions**, toggle **Account Permissions**, traverse to **Models** and enable only the read-access required for GitHub Models.


Copy your new token that you have just created. You will now add this to your `.env` file included in this project.

## Retrieve an OpenCage API Key ```GEO_API_KEY```

This project leverages geocoding information obtained from OpenCage's Geocoder API endpoint.

To use OpenCage, you will need to create an [OpenCage User Profile](https://opencagedata.com/)

1. Sign up for a free account (or log in if you already have one).

2. Once logged in, go to your dashboard.

3. Click “Create API Key” or “Generate API Key”.

4. Copy the generated key.

5. Add it to your .env file like this:

   ```GEO_API_KEY=your_opencage_api_key```

## Retrieve an OpenWeather API Key ```OPEN_WEATHER_API_KEY```

This project leverages weather forecast data obtained from OpenWeather’s API.

To use OpenWeather, you will need to create an [OpenWeather Account](https://openweathermap.org/api)

1. Sign up for a free account (or log in if you already have one).

2. Once logged in, select **API** from the top bar and click **Subscribe**. This will prompt you to begin an OpenWeather subscription plan. The **Free tier** is sufficient for this project.

3. Go to your **API Keys** section in your profile/dashboard.

4. Click **“Create Key”** or **“Generate API Key”**.

5. Copy the generated key.

6. Add it to your `.env` file like this:

   ```OPEN_WEATHER_API_KEY=your_openweather_api_key```
   
# Running AgroAskAI

From the root directory, run:

```python main.py```

You will be prompted to enter a user input and, if needed, additional clarification.



