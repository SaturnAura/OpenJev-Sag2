from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from scripts.serve import create_app


class FakeEngine:
    config = SimpleNamespace(model_path="fake-9b")
    readout = SimpleNamespace(name="slot_head")

    def decide(self, state, questions):
        assert state == "passage"
        assert "relevant" in questions
        return SimpleNamespace(
            to_dict=lambda: {
                "results": {
                    "relevant": {
                        "probabilities": {"true": 0.75, "false": 0.25},
                        "answer": True,
                    }
                },
                "usage": {},
                "timing": {},
            }
        )


def test_decide_endpoint_returns_native_probabilities():
    client = TestClient(create_app(FakeEngine()))
    response = client.post(
        "/v1/decide",
        json={
            "state": "passage",
            "questions": {
                "relevant": {
                    "type": "bool",
                    "instructions": "Is this passage relevant?",
                    "criteria": {"true": "Yes", "false": "No"},
                }
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["results"]["relevant"]["probabilities"]["true"] == 0.75

