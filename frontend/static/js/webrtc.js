// WebRTC helpers for the VoiceGuard demo.
// Primary: room-based auto-signaling via the backend (/ws/signal/<room>).
// Fallback: manual SDP copy-paste (VG.encodeSdp / VG.decodeSdp), still exported.

window.VG = window.VG || {};

VG.createPeer = function () {
  return new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });
};

VG.waitForIce = function (pc) {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 3000); // don't wait forever on flaky networks
  });
};

VG.encodeSdp = (desc) => btoa(JSON.stringify(desc));
VG.decodeSdp = (text) => new RTCSessionDescription(JSON.parse(atob(text.trim())));

// role: "caller" (creates offer) | "receiver" (answers). Resolves when connected.
VG.autoConnect = function (role, room, pc, onStatus = () => {}) {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/signal/${encodeURIComponent(room)}`);
    const sendSignal = (o) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify(o));

    pc.onicecandidate = (e) => {
      if (e.candidate) sendSignal({ type: "ice", candidate: e.candidate });
    };
    pc.onconnectionstatechange = () => {
      onStatus("peer: " + pc.connectionState);
      if (pc.connectionState === "connected") resolve(ws);
      if (["failed", "closed"].includes(pc.connectionState)) reject(new Error(pc.connectionState));
    };

    ws.onerror = () => reject(new Error("signaling failed"));
    ws.onmessage = async (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "full") return reject(new Error("room is full"));
      if (m.type === "joined") return onStatus(`joined as ${m.role} — waiting for peer`);
      if (m.type === "peer-left") return onStatus("peer left");

      if (m.type === "ready" && role === "caller") {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        sendSignal({ type: "sdp", sdp: pc.localDescription });
        onStatus("offer sent");
      } else if (m.type === "sdp") {
        await pc.setRemoteDescription(m.sdp);
        if (m.sdp.type === "offer") {
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          sendSignal({ type: "sdp", sdp: pc.localDescription });
          onStatus("answer sent");
        }
      } else if (m.type === "ice") {
        try { await pc.addIceCandidate(m.candidate); } catch (_) {}
      }
    };
  });
};
