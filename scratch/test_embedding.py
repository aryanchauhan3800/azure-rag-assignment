import os
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

api_key = os.environ.get("AZURE_OPENAI_API_KEY")
endpoint_full = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Test 1: Using OpenAI client with base_url
print("--- Test 1: Standard OpenAI Client ---")
try:
    client1 = OpenAI(base_url=endpoint_full, api_key=api_key)
    res1 = client1.embeddings.create(input="test embedding", model=deployment)
    print(f"Success! Dimension: {len(res1.data[0].embedding)}")
except Exception as e:
    print(f"Test 1 Failed: {e}")

# Test 2: Using AzureOpenAI client with stripped endpoint
print("--- Test 2: AzureOpenAI Client (stripped endpoint) ---")
try:
    stripped_endpoint = endpoint_full.replace("/openai/v1", "").replace("/v1", "")
    if not stripped_endpoint.endswith("/"):
        stripped_endpoint += "/"
        
    client2 = AzureOpenAI(
        azure_endpoint=stripped_endpoint,
        api_key=api_key,
        api_version=api_version
    )
    res2 = client2.embeddings.create(input="test embedding", model=deployment)
    print(f"Success! Dimension: {len(res2.data[0].embedding)}")
except Exception as e:
    print(f"Test 2 Failed: {e}")

