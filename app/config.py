import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(Path(__file__).resolve().with_name(".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables.")

PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")


def project_path(name: str, default: str) -> str:
    path = Path(os.getenv(name, default))
    return str(path if path.is_absolute() else PROJECT_DIR / path)


DOCS_DIR = project_path("DOCS_DIR", "campus_docs")
INDEX_DIR = project_path("INDEX_DIR", "faiss_index")