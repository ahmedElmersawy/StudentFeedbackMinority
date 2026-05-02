import os

import reflex as rx

_api_url = os.environ.get("API_URL", "").strip()

_plugins = [
    rx.plugins.SitemapPlugin(),
    rx.plugins.TailwindV4Plugin(),
]

# When unset, Reflex uses the default local backend URL. For public deploys set
# API_URL to your reachable backend, e.g. https://api.example.com:8000
config = (
    rx.Config(app_name="feedback_web", api_url=_api_url, plugins=_plugins)
    if _api_url
    else rx.Config(app_name="feedback_web", plugins=_plugins)
)
