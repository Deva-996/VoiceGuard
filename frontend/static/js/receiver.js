// Receiver tab: join by room code, capture the remote stream, stream PCM to the backend.
(() => {
  const $ = (id) => document.getElementById(id);
  const setStatus = (s) => ($("status").textContent = s);
  let pc, ws, ctx;

  $("answer").onclick = async () => {
    try {
      const room = ($("room").value || "").trim();
      if (!room) return setStatus("enter the same room code as the caller");
      $("answer").disabled = true;

      pc = VG.createPeer();
      pc.ontrack = (ev) => {
        $("remoteAudio").srcObject = ev.streams[0];
        startStreaming(ev.streams[0]);
      };

      setStatus("connecting…");
      await VG.autoConnect("receiver", room, pc, setStatus);
      setStatus("connected — analysing incoming audio");
    } catch (e) {
      setStatus("error: " + e.message);
      $("answer").disabled = false;
    }
  };

  async function startStreaming(remoteStream) {
    await fetch("/config").then((r) => r.json()).catch(() => ({}));

    ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(remoteStream);

    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () =>
      ws.send(JSON.stringify({ type: "config", input_sample_rate: ctx.sampleRate }));
    ws.onmessage = (ev) => VG.dashboard.handle(JSON.parse(ev.data));
    ws.onclose = () => setStatus("ws closed");
    ws.onerror = () => setStatus("ws error");

    const send = (buf) => ws.readyState === WebSocket.OPEN && ws.send(buf);

    try {
      await ctx.audioWorklet.addModule("/static/js/pcm-worklet.js");
      const node = new AudioWorkletNode(ctx, "pcm-worklet", {
        processorOptions: { frameMs: 85 },
      });
      node.port.onmessage = (e) => send(e.data);
      src.connect(node);
      const sink = ctx.createGain();
      sink.gain.value = 0; // don't echo the caller back out
      node.connect(sink).connect(ctx.destination);
      setStatus(`streaming @ ${ctx.sampleRate} Hz → backend (AudioWorklet)`);
    } catch (e) {
      const node = ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (ev) => {
        const f32 = ev.inputBuffer.getChannelData(0);
        const i16 = new Int16Array(f32.length);
        for (let i = 0; i < f32.length; i++) {
          const s = Math.max(-1, Math.min(1, f32[i]));
          i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        send(i16.buffer);
      };
      src.connect(node);
      node.connect(ctx.destination);
      setStatus(`streaming @ ${ctx.sampleRate} Hz → backend (ScriptProcessor fallback)`);
    }
  }

  window.addEventListener("beforeunload", () => {
    try { ws && ws.send(JSON.stringify({ type: "end" })); } catch (_) {}
    try { ctx && ctx.close(); } catch (_) {}
  });
})();
