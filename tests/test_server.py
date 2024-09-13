import pytest
import requests
import subprocess
import re
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import wait_for_server_ready, load_config

# TODO: CREATE A TEST TO CHECK FOR OCCURENCE OF STRING "NVIDIA" IF PROMTED ABOUT NVIDIA


config = load_config(config_path="config.json")
BASE_URL = config.get("base_url", "")
ENDPOINT = config.get("endpoint", "")


def test_answer_contains_no_unwanted_sections() -> None:
    """
    Test to ensure the answer does not contain any unwanted sections by asking a random question.
    """

    query = {"query": "Where is Latvia located?"}
    response = requests.post(f"{BASE_URL}{ENDPOINT}", json=query)
    assert response.status_code == 200
    data = response.json()

    pattern = re.compile(r"\b[A-Z][A-Za-z\s]+:\b")
    matches = pattern.findall(data["answer"])
    assert not any(match != "Answer:" for match in matches)


def test_valid_query() -> None:
    """
    Test a valid query to ensure the API returns a valid response.
    """

    query = {"query": "What is NVIDIA known for?"}
    response = requests.post(f"{BASE_URL}{ENDPOINT}", json=query)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_missing_prompt() -> None:
    """
    Test a request with a missing query parameter.
    """

    response = requests.post(f"{BASE_URL}{ENDPOINT}", json={})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Query parameter is required"


def test_empty_query() -> None:
    """
    Test a request with an empty query string.
    """

    query = {"query": ""}
    response = requests.post(f"{BASE_URL}{ENDPOINT}", json=query)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Query parameter is required"


def test_invalid_endpoint() -> None:
    """

    Test an invalid endpoint to ensure the server handles it gracefully.
    """
    response = requests.get(f"{BASE_URL}/invalid")
    assert response.status_code == 404


def test_large_query() -> None:
    """
    Test a very large query to check if the server handles it correctly.
    """

    large_query = {"query": "x" * 10000}
    response = requests.post(f"{BASE_URL}{ENDPOINT}", json=large_query)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)


@pytest.fixture(scope="module", autouse=True)
def setup_server():
    """
    Setup and teardown of the server for testing.
    """

    global server_process

    # Start the server
    server_process = subprocess.Popen(
        ["python", "server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait until the server is ready
    if not wait_for_server_ready(url=f"{BASE_URL}/health"):
        raise RuntimeError("Server did not start within the timeout period.")

    # Yield control to the tests
    yield

    # Teardown: Stop the server
    server_process.terminate()
    server_process.wait()
