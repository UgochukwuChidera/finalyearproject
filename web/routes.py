"""Route definitions — thin module that imports all sub-modules.

Sub-modules register their routes on the shared blueprint defined in .bp.
"""
from .bp import bp
from . import job_routes, config_routes, api_routes, review_routes
