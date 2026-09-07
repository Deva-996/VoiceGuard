// Live risk gauge + alert banner + event log.
window.VG = window.VG || {};

VG.dashboard = (() => {
  const $ = (id) => document.getElementById(id);
  let lastAlert = 0;
  let dropped = 0;

  function handle(msg) {
    if (msg.type === "ready") {
      log(`session ${msg.session_id} — waiting for audio`);
      return;
    }
    if (msg.type !== "score") return;

    const { score, level } = msg.risk;
    const g = $("gauge");
    g.dataset.level = level;
    g.style.setProperty("--fill", `${Math.round(score * 100)}%`);
    $("gaugeScore").textContent = score.toFixed(2);
    $("gaugeLevel").textContent = level;

    const isHigh = level === "HIGH";
    $("alertBanner").hidden = !isHigh;
    if (isHigh && msg.alert) lastAlert = msg.index;

    dropped += msg.dropped || 0;
    const behind = msg.dropped ? `  ⋯${msg.dropped}` : "";
    log(
      `#${String(msg.index).padStart(3)}  chunk ${msg.fake_prob.toFixed(3)}  ` +
        `roll ${score.toFixed(3)}  ${level}${msg.alert ? "  ⚠ ALERT" : ""}` +
        `  ${Math.round(msg.latency_ms)}ms${behind}`
    );
    if (dropped && $("meta")) $("meta").textContent = `${dropped} windows dropped (backend catching up)`;
  }

  function log(line) {
    const li = document.createElement("li");
    li.textContent = line;
    const el = $("log");
    el.prepend(li);
    while (el.childElementCount > 120) el.lastElementChild.remove();
  }

  return { handle, log };
})();
