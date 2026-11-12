"""MADRAJ multi-agent airport turnaround system."""
from pathlib import Path

# Load the keys from the .env file in the project folder (if it exists)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Turn tracing off if no real LangSmith key was added yet (avoids 403 errors)
import os as _os
if not _os.getenv("LANGSMITH_API_KEY", "").startswith("lsv2_"):
    _os.environ["LANGSMITH_TRACING"] = "false"
