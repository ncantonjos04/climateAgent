Set Up for Samples using GitHub Models
Step 1: Retrieve Your GitHub Personal Access Token (PAT)
This course leverages the GitHub Models Marketplace, providing free access to Large Language Models (LLMs) that you will use to build AI Agents.

To use the GitHub Models, you will need to create a GitHub Personal Access Token.

This can be done by going to your Personal Access Tokens settings in your GitHub Account.

Please follow the Principle of Least Privilege when creating your token. This means you should only give the token the permissions it needs to run the code samples in this course.

Select the Fine-grained tokens option on the left side of your screen.

Then select Generate new token.

Generate Token

Enter a descriptive name for your token that reflects its purpose, making it easy to identify later. Set an expiration date (recommended: 30 days; you can choose a shorter period like 7 days if you prefer a more secure posture.)

Token Name and Expiration

Limit the token's scope to your fork of this repository.

Limit scope to fork repository

Restrict the token's permissions: Under Permissions, toggle Account Permissions, traverse to Models and enable only the read-access required for GitHub Models.

Account Permissions

Models Read Access

Copy your new token that you have just created. You will now add this to your .env file included in this course.