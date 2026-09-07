// Receiver tab: answer the call, capture the remote stream, stream PCM to the backend. [Day 6]
(() => {
  const $ = (id) => document.getElementById(id);
  const setStatus = (s) => ($("status").textContent = s);
  let pc;

  $("answer").onclick = async () => {
    pc = VG.createPeer();
    pc.ontrack = (ev) => {
      $("remoteAudio").srcObject = ev.streams[0];
      startStreaming(ev.streams[0]);
    };
    pc.onconnectionstatechange = () => setStatus("peer: " + pc.connectionState);

    await pc.setRemoteDescription(VG.decodeSdp($("remoteSdp").value));
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await VG.waitForIce(pc);
    $("localSdp").value = VG.encodeSdp(pc.localDescription);
    setStatus("answer ready — send it back to the caller");
  };

  async function startStreaming(remoteStream) {
    const cfg = await fetch("/config").then((r) => r.json());
    const targetSr = cfg.audio.sample_rate;

    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(remoteStream);

    // AudioWorklet is the right tool; ScriptProcessor kept here for a dependency-free skeleton.
    const node = ctx.createScriptProcessor(4096, 1, 1);
    const ws = new WebSocket(`ws://${location.host}/ws/stream`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => ws.send(JSON.stringify({ type: "config", input_sample_rate: ctx.sampleRate }));
    ws.onmessage = (ev) => VG.dashboard.handle(JSON.parse(ev.data));
    ws.onclose = () => setStatus("ws closed");

    node.onaudioprocess = (e) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      ws.send(i16.buffer);
    };

    src.connect(node);
    node.connect(ctx.destination);
    setStatus(`streaming @ ${ctx.sampleRate} Hz -> backend (target ${targetSr} Hz)`);
  }
})();
