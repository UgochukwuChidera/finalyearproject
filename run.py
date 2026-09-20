import os

from web import create_app
from dotenv import load_dotenv

load_dotenv()
app = create_app()

# Production: gunicorn -w 2 -k gthread --threads 5 -b 127.0.0.1:8000 'web:create_app()'
# Dev only: python run.py (Flask dev server, single-process)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=os.getenv("FLASK_DEBUG", "0") == "1")
