// Minimal manual-signaling WebRTC helper for the VoiceGuard demo. [Day 6]
// Manual copy-paste of SDP keeps the backend free of a signaling server for the demo;
// swap for a /ws/signal endpoint later.

window.VG = window.VG || {};

VG.createPeer = function () {
  return new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });
};

// Resolve once ICE gathering is complete so the SDP we print is final.
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
  });
};

VG.encodeSdp = (desc) => btoa(JSON.stringify(desc));
VG.decodeSdp = (text) => new RTCSessionDescription(JSON.parse(atob(text.trim())));
