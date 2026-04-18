from pathlib import Path

from flask import Flask

from ai_extraction.dictionary_matcher import DictionaryStore


def create_app(root_dir: str | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    base = Path(root_dir or Path(__file__).resolve().parents[1])

    app.config["ROOT_DIR"] = str(base)
    app.config["CONFIGS_DIR"] = str(base / "configs")
    app.config["DICTIONARIES_DIR"] = str(base / "dictionaries")
    app.config["UPLOADS_DIR"] = str(base / "uploads")
    app.config["OUTPUTS_DIR"] = str(base / "outputs")
    app.config["LOGS_DIR"] = str(base / "logs")

    for key in ["CONFIGS_DIR", "DICTIONARIES_DIR", "UPLOADS_DIR", "OUTPUTS_DIR", "LOGS_DIR"]:
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)

    store = DictionaryStore(app.config["DICTIONARIES_DIR"])
    app.config["DICTIONARY_STORE"] = store
    app.config["DICTIONARIES_CACHE"] = store.load()

    from .routes import bp

    app.register_blueprint(bp)
    return app
