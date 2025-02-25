import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Load data from CSV file
CSV_FILE_PATH = 'data.csv'  # Replace with your CSV file path
df = pd.read_csv(CSV_FILE_PATH)

# Display sample data
print("Sample Data:")
print(df.head())

# Convert data to string format for prompt
data_str = df.to_string(index=False)

# Define the query
query = input()

# Construct the prompt for the AI model
prompt = f"""
You are an assistant with access to logistics data. 
Answer questions based on the provided data.

Data:
{data_str}

Question:
{query}

Answer:
"""

# Invoke the model with the constructed prompt
result = model.invoke(prompt)

# Output the AI's response
print("\nResponse from Gemini:")
print(result.content)
