from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from docqa.api import create_app
from docqa.config import Settings
from docqa.llm import Answer, Citation, ClaudeAnswerer

SAMPLE_DOCS = Path(__file__).parent.parent / "sample_docs"


class FakeAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, question, chunks):
        self.calls.append((question, chunks))
        top = chunks[0]
        citation = Citation(top.id, top.source, top.text[:40])
        return Answer(f"See {top.source} [1]", [citation], "fake")


@pytest.fixture
def answerer():
    return FakeAnswerer()


@pytest.fixture
def client(tmp_path, answerer):
    settings = Settings(index_path=tmp_path / "index.json", seed_dir=SAMPLE_DOCS)
    return TestClient(create_app(settings, answerer))


def test_health_and_seeded_documents(client):
    assert client.get("/health").json()["chunks"] > 0
    assert set(client.get("/documents").json()) == {
        "employee-handbook.md",
        "onboarding-guide.md",
        "security-policy.md",
    }


def test_ask_returns_answer_citations_and_retrieval(client, answerer):
    res = client.post("/ask", json={"question": "minimum password length", "top_k": 2})
    assert res.status_code == 200
    body = res.json()
    assert body["citations"][0]["n"] == 1
    assert body["citations"][0]["source"] == "security-policy.md"
    assert len(body["retrieved"]) <= 2
    assert len(answerer.calls[0][1]) <= 2


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "top_k": 99}).status_code == 422


def test_upload_then_ask_then_delete(client):
    files = {"files": ("cafeteria.txt", b"The cafeteria serves lunch from noon.", "text/plain")}
    assert client.post("/documents", files=files).json() == {"cafeteria.txt": 1}
    body = client.post("/ask", json={"question": "when is the cafeteria lunch"}).json()
    assert body["retrieved"][0]["source"] == "cafeteria.txt"

    assert client.delete("/documents/cafeteria.txt").json() == {"removed_chunks": 1}
    assert client.delete("/documents/cafeteria.txt").status_code == 404


def test_upload_rejects_unsupported_type(client):
    files = {"files": ("image.png", b"\x89PNG", "image/png")}
    assert client.post("/documents", files=files).status_code == 415


def test_index_persists_across_restarts(tmp_path, answerer):
    settings = Settings(index_path=tmp_path / "index.json")
    first = TestClient(create_app(settings, answerer))
    files = {"files": ("notes.md", b"Quarterly planning starts in January.", "text/markdown")}
    first.post("/documents", files=files)

    second = TestClient(create_app(settings, answerer))
    assert second.get("/documents").json() == {"notes.md": 1}


def test_missing_api_key_gives_clear_503(tmp_path):
    def create(**_):  # what the Anthropic SDK raises when no credentials are configured
        raise TypeError('"Could not resolve authentication method. Expected one of api_key"')

    sdk = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    settings = Settings(index_path=tmp_path / "index.json", seed_dir=SAMPLE_DOCS)
    app = create_app(settings, ClaudeAnswerer("claude-opus-5-5", client=sdk))
    res = TestClient(app).post("/ask", json={"question": "How many PTO days?"})
    assert res.status_code == 503
    assert "ANTHROPIC_API_KEY" in res.json()["detail"]


def test_ui_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "DocQA" in res.text
