// Caller tab: capture mic, connect to the receiver by room code.
(() => {
  const $ = (id) => document.getElementById(id);
  const setStatus = (s) => ($("status").textContent = s);
  let pc, stream;

  $("start").onclick = async () => {
    try {
      const room = ($("room").value || "").trim();
      if (!room) return setStatus("enter a room code");
      $("start").disabled = true;

      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      pc = VG.createPeer();
      stream.getTracks().forEach((t) => pc.addTrack(t, stream));

      setStatus("connecting…");
      await VG.autoConnect("caller", room, pc, setStatus);
      setStatus("connected — talk (or play a clip) into the mic");
    } catch (e) {
      setStatus("error: " + e.message);
      $("start").disabled = false;
    }
  };
})();
