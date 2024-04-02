"""
For deploying a ASGI application
"""

import os

from django.core.asgi import get_asgi_application
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myeachtra.settings")

myeachtra = get_asgi_application()
