// Live risk gauge + alert banner + event log. [Day 6]
window.VG = window.VG || {};

VG.dashboard = (() => {
  const $ = (id) => document.getElementById(id);

  function handle(msg) {
    if (msg.type === "ready") {
      log(`session ${msg.session_id}`);
      return;
    }
    if (msg.type !== "score") return;

    const { score, level } = msg.risk;
    $("gauge").dataset.level = level;
    $("gaugeScore").textContent = score.toFixed(2);
    $("gaugeLevel").textContent = level;
    $("alertBanner").hidden = !(level === "HIGH");

    log(
      `#${msg.index}  chunk=${msg.fake_prob.toFixed(3)}  roll=${score.toFixed(3)}  ` +
        `${level}${msg.alert ? "  ⚠ ALERT" : ""}  (${msg.latency_ms} ms)`
    );
  }

  function log(line) {
    const li = document.createElement("li");
    li.textContent = line;
    const el = $("log");
    el.prepend(li);
    while (el.childElementCount > 100) el.lastElementChild.remove();
  }

  return { handle, log };
})();
