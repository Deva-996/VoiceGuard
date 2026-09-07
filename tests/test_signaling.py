"""WS /ws/signal/<room> — the room-based WebRTC signaling relay."""

import json

from fastapi.testclient import TestClient

from backend.main import app


def test_two_peers_join_and_relay():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/signal/r1") as a:
            assert a.receive_json() == {"type": "joined", "role": "caller"}
            with client.websocket_connect("/ws/signal/r1") as b:
                assert b.receive_json() == {"type": "joined", "role": "receiver"}
                # both told "ready" once the room has 2
                assert a.receive_json()["type"] == "ready"
                assert b.receive_json()["type"] == "ready"

                a.send_text(json.dumps({"type": "sdp", "sdp": {"type": "offer"}}))
                relayed = b.receive_json()
                assert relayed["type"] == "sdp" and relayed["sdp"]["type"] == "offer"

                b.send_text(json.dumps({"type": "ice", "candidate": "x"}))
                assert a.receive_json() == {"type": "ice", "candidate": "x"}


def test_third_peer_rejected():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/signal/r2") as a, \
             client.websocket_connect("/ws/signal/r2") as b:
            a.receive_json(); b.receive_json(); a.receive_json(); b.receive_json()
            with client.websocket_connect("/ws/signal/r2") as c:
                assert c.receive_json() == {"type": "full"}
