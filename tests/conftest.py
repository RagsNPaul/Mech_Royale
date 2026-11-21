"""Shared fixtures for the recorded stream and the rules snapshot."""

import pytest

from bot.config import Rules
from tests.fixtures import load_stream


@pytest.fixture(scope="session")
def hello():
    message, _observations = load_stream()
    return message


@pytest.fixture(scope="session")
def observations():
    _hello, messages = load_stream()
    return messages


@pytest.fixture(scope="session")
def rules(hello):
    return Rules(hello["config"])
