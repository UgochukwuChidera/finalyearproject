"""Shared Blueprint instance for the web module."""
from flask import Blueprint

bp = Blueprint("web", __name__)
