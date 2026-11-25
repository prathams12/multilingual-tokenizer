from pathlib import Path
from dotenv import load_dotenv
import os
from huggingface_hub import HfApi

load_dotenv(Path(".env"))

user = HfApi().whoami(token=os.getenv("HF_TOKEN"))
print(f"Authenticated as: {user['name']}")