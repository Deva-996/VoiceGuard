// Caller tab: capture mic, offer to receiver. [Day 6]
(() => {
  const $ = (id) => document.getElementById(id);
  const setStatus = (s) => ($("status").textContent = s);
  let pc, stream;

  $("start").onclick = async () => {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    pc = VG.createPeer();
    stream.getTracks().forEach((t) => pc.addTrack(t, stream));
    pc.onconnectionstatechange = () => setStatus("peer: " + pc.connectionState);

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await VG.waitForIce(pc);
    $("localSdp").value = VG.encodeSdp(pc.localDescription);
    $("connect").disabled = false;
    setStatus("offer ready — send it to the receiver");
  };

  $("connect").onclick = async () => {
    await pc.setRemoteDescription(VG.decodeSdp($("remoteSdp").value));
    setStatus("connected");
  };
})();
