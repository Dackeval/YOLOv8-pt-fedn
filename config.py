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
    "DISCOVER_HOST": "api.fedn.scaleoutsystems.com/test-zxw-fedn-reducer",
    "DATA_PATH": "/home/nviduser/YOLOv8-pt-fedn/dataset",
    "DISCOVER_PORT": None,
    "IS_LOCAL": False,
    "SECURE": True,
    "VERIFY": True,
    "CLIENT_TOKEN": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzU4OTgxMzQ0LCJpYXQiOjE3NTYzODkzNDQsImp0aSI6IjA2Mzk4NTI0ZTc1MTQ5YmI5ZDJmZDczMWZkYTNjYzA0IiwidXNlcl9pZCI6MTYxLCJjcmVhdG9yIjoiU2NhbGVvdXRpbnRlcm4iLCJyb2xlIjoiY2xpZW50IiwicHJvamVjdF9zbHVnIjoidGVzdC16eHcifQ.7AszTEa4NWkl7be4OeMpny-AwI3HOlGiUZZ_UqOIJf0",
    "ADMIN_TOKEN": None,
}

# Choose which environment to use
USE_LOCAL = False  # Set to False to use remote environment

# Combine the selected environment config with common settings
settings = {**(LOCAL_CONFIG if USE_LOCAL else REMOTE_CONFIG)}