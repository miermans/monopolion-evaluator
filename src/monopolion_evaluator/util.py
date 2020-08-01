import datetime
import os
import re


def get_model_path(path: str):
    if path is not None:
        if os.path.isdir(path):
            date_str = re.sub(':', '', datetime.datetime.now().replace(microsecond=0).isoformat())
            path = os.path.join(path, date_str)

    return path


def validate_write_access(path: str):
    with open(path, "w") as f:
        pass
