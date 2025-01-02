from promptbuilder.llm_client import LLMClient
import os
import dotenv

dotenv.load_dotenv()
model = "fireworks:fireworks/llama-v3p3-70b-instruct"
api_key = os.getenv('FIREWORKS_API_KEY')
llm_client = LLMClient(model=model, api_key=api_key)
print(llm_client.make_request("What is the capital of France?"))
