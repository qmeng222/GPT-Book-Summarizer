import openai
from dotenv import load_dotenv
import os

load_dotenv(".env")
openai.api_key = os.environ["OPENAI_API_KEY"]
