import pytest
import re

from monopolion_evaluator.util import get_model_path
from monopolion_evaluator.util import validate_write_access


FAKE_MODEL_DIR = "/tmp/foo"
FAKE_MODEL_FILE = f"{FAKE_MODEL_DIR}/bar"


def test_get_model_path_from_full_path(fs):
    fs.create_file(FAKE_MODEL_FILE)
    assert FAKE_MODEL_FILE == get_model_path(FAKE_MODEL_FILE)


def test_get_model_path_from_dir(fs):
    fs.create_dir(FAKE_MODEL_DIR)
    assert re.match(FAKE_MODEL_DIR + r'/\d{4}-\d{2}-\d{2}T\d{6}', get_model_path(FAKE_MODEL_DIR))


def test_validate_model_path(fs):
    fs.create_file(FAKE_MODEL_FILE)
    validate_write_access(FAKE_MODEL_FILE)


def test_validate_write_access_permission_error(fs):
    fs.create_dir(FAKE_MODEL_DIR, perm_bits=0o000)
    path = get_model_path(FAKE_MODEL_DIR)

    with pytest.raises(PermissionError):
        validate_write_access(path)
