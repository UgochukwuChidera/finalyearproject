from web import create_app
from dotenv import load_dotenv

load_dotenv()
app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
