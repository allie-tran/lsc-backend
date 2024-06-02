from fastapi import FastAPI
from fastapi.testclient import TestClient
from .main import app
import pytest
import json

client = TestClient(app)

def test_request_search():
    data = {"main": "what is the name of the salon I own?",
            "size": 5,
            "session_id": "test"}
    response = client.post("/search", json=data)
    assert response.status_code == 200
    assert "searchToken" in response.json()

    token = response.json()["searchToken"]
    done = {
        "raw": False,
        "answers": False,
        "modified": False
    }
    with client.stream("GET", f"/get-stream-results/test/{token}") as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        for text in response.iter_lines():
            # Remove data: from the beginning of the line
            text = text.replace("data: ", "").strip()
            if not text:
                continue
            if text == "END":
                print("End of stream")
                break
            else:
                data = json.loads(text)
                print("OK")
                if data["type"] == "raw":
                    done["raw"] = True
                if data["type"] == "answers":
                    done["answers"] = True
                if data["type"] == "modified":
                    done["modified"] = True
    assert all(done.values())



