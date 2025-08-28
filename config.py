import json
from pathlib import Path

# Environment configurations
LOCAL_CONFIG = {
    "DISCOVER_HOST": "127.0.0.1",
    "DISCOVER_PORT": 8092,
    "IS_LOCAL": True,
    "SECURE": False,
    "VERIFY": False,
    "CLIENT_TOKEN": None,
    "ADMIN_TOKEN": None,
}

REMOTE_CONFIG = {
    "DISCOVER_HOST": "api.studio.scaleoutplatform.com/yolo-wwy-fedn-reducer",
    "DISCOVER_PORT": None,
    "IS_LOCAL": False,
    "SECURE": True,
    "VERIFY": True,
    "CLIENT_TOKEN": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzU4OTczOTgzLCJpYXQiOjE3NTYzODE5ODMsImp0aSI6IjljZDFkOTYzZGZlYTQ1YmI4NTUxNzY5MTdjMWQzOWZjIiwidXNlcl9pZCI6IjU4IiwiY3JlYXRvciI6InNpZ3ZhcmRAc2NhbGVvdXRzeXN0ZW1zLmNvbSIsInJvbGUiOiJjbGllbnQiLCJwcm9qZWN0X3NsdWciOiJ5b2xvLXd3eSJ9.IFdafa--xq81mp9H3a2W_1mPop_B02becEkN-gG81wE",
    "ADMIN_TOKEN": None,
}

# Choose which environment to use
USE_LOCAL = False  # Set to False to use remote environment

# Combine the selected environment config with common settings
settings = {**(LOCAL_CONFIG if USE_LOCAL else REMOTE_CONFIG)}