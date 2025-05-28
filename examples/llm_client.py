from promptbuilder.llm_client import get_client
import os
import dotenv

dotenv.load_dotenv()
model = "openai:gpt-4o-mini"
api_key = os.getenv('OPENAI_API_KEY')
llm_client = get_client(model=model, api_key=api_key)
print(llm_client.from_text("What is the capital of France?"))
