import os
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

api_key = os.environ.get("AZURE_OPENAI_API_KEY")
endpoint_full = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

client = None
if "/openai/v1" in endpoint_full:
    client = OpenAI(base_url=endpoint_full, api_key=api_key)
else:
    client = AzureOpenAI(azure_endpoint=endpoint_full, api_key=api_key, api_version=api_version)

models_to_test = ["gpt-4o", "gpt-4o-mini", "gpt-35-turbo", "gpt-4"]

for model in models_to_test:
    print(f"Testing {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=10
        )
        print(f"Success! Model {model} is available.")
        break
    except Exception as e:
        print(f"Failed: {e}")
