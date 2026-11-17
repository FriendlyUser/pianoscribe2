(() => {
  "use strict";
  const notes = JSON.parse(document.getElementById("playback-data").textContent)
    .map(([id, pitch, start, duration]) => ({ id, pitch, start, duration }))
    .sort((a, b) => a.start - b.start);
  const total = notes.reduce((max, n) => Math.max(max, n.start + n.duration), 0);
  const playBtn = document.getElementById("play");
  const stopBtn = document.getElementById("stop");
  const speedInput = document.getElementById("speed");
  const speedOut = document.getElementById("speed-out");
  const clock = document.getElementById("clock");
  if (new URLSearchParams(location.search).has("embed")) document.body.classList.add("embed");

  let ctx = null, master = null;
  let playing = false, position = 0, anchorCtx = 0, anchorPos = 0, speed = 1;
  let nextIndex = 0, timer = null, raf = null;
  const lit = new Map();
  const voices = new Set();

  const elementsFor = (id) => [document.getElementById(id), document.getElementById("c-" + id)].filter(Boolean);
  const fmt = (ms) => { const s = Math.max(0, Math.floor(ms / 1000)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
  const now = () => (playing ? anchorPos + (ctx.currentTime - anchorCtx) * 1000 * speed : position);

  function ensureAudio() {
    if (ctx) return;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    master = ctx.createGain();
    master.gain.value = 0.5;
    const comp = ctx.createDynamicsCompressor();
    master.connect(comp).connect(ctx.destination);
  }

  // A small additive "piano-like" voice: fast attack, exponential decay, short release.
  function voice(pitch, when, seconds) {
    const freq = 440 * Math.pow(2, (pitch - 69) / 12);
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = Math.min(9000, freq * 6 + 600);
    const level = 0.22 * Math.pow(0.985, Math.max(0, pitch - 60));
    const hold = Math.max(0.08, seconds);
    gain.gain.setValueAtTime(0.0001, when);
    gain.gain.exponentialRampToValueAtTime(level, when + 0.006);
    gain.gain.exponentialRampToValueAtTime(level * 0.25, when + Math.min(hold, 1.6));
    gain.gain.setValueAtTime(level * 0.25, when + hold);
    gain.gain.exponentialRampToValueAtTime(0.0001, when + hold + 0.18);
    const oscs = [["triangle", 1, 1], ["sine", 2, 0.35], ["sine", 3, 0.12]].map(([type, mult, amp]) => {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = type; osc.frequency.value = freq * mult; g.gain.value = amp;
      osc.connect(g).connect(filter);
      osc.start(when); osc.stop(when + hold + 0.25);
      return osc;
    });
    filter.connect(gain).connect(master);
    const handle = { gain, oscs };
    voices.add(handle);
    oscs[0].onended = () => voices.delete(handle);
  }

  function schedule() {
    const horizon = now() + 250 * speed;
    while (nextIndex < notes.length && notes[nextIndex].start <= horizon) {
      const n = notes[nextIndex++];
      const when = anchorCtx + (n.start - anchorPos) / 1000 / speed;
      voice(n.pitch, Math.max(ctx.currentTime, when), n.duration / 1000 / speed);
    }
  }

  function paint() {
    const t = now();
    clock.textContent = fmt(t) + " / " + fmt(total);
    for (const [id, end] of lit) {
      if (t >= end || t < 0) { elementsFor(id).forEach((el) => el.classList.remove("playing")); lit.delete(id); }
    }
    let first = null;
    // At rest on the first beat nothing is sounding yet, so nothing is lit.
    const idle = !playing && position === 0;
    for (const n of idle ? [] : notes) {
      if (n.start > t) break;
      if (t < n.start + n.duration && !lit.has(n.id)) {
        lit.set(n.id, n.start + n.duration);
        const els = elementsFor(n.id);
        els.forEach((el) => el.classList.add("playing"));
        first = first || els.find((el) => el.getClientRects().length);
      }
    }
    if (first && playing) {
      const rect = first.getBoundingClientRect();
      if (rect.bottom > window.innerHeight - 80 || rect.top < 90) {
        window.scrollBy({ top: rect.top - window.innerHeight * 0.3, behavior: "smooth" });
      }
    }
    if (playing && t >= total + 300) { stop(); return; }
    if (playing) raf = requestAnimationFrame(paint);
  }

  function silence() {
    voices.forEach(({ gain, oscs }) => {
      try { gain.gain.cancelScheduledValues(ctx.currentTime); gain.gain.setTargetAtTime(0.0001, ctx.currentTime, 0.02); oscs.forEach((o) => o.stop(ctx.currentTime + 0.1)); } catch (e) { /* already stopped */ }
    });
    voices.clear();
  }

  function clearLit() {
    lit.forEach((_, id) => elementsFor(id).forEach((el) => el.classList.remove("playing")));
    lit.clear();
  }

  function play(from) {
    ensureAudio();
    ctx.resume();
    position = from >= total ? 0 : from;
    anchorCtx = ctx.currentTime + 0.08;
    anchorPos = position;
    nextIndex = notes.findIndex((n) => n.start >= position);
    if (nextIndex < 0) nextIndex = notes.length;
    playing = true;
    playBtn.textContent = "Pause";
    playBtn.setAttribute("aria-pressed", "true");
    schedule();
    timer = setInterval(schedule, 60);
    raf = requestAnimationFrame(paint);
  }

  function pause() {
    position = now();
    playing = false;
    clearInterval(timer); cancelAnimationFrame(raf);
    if (ctx) silence();
    playBtn.textContent = "Play";
    playBtn.setAttribute("aria-pressed", "false");
  }

  function stop() { pause(); position = 0; clearLit(); paint(); }

  playBtn.addEventListener("click", () => (playing ? pause() : play(position)));
  stopBtn.addEventListener("click", stop);
  speedInput.addEventListener("input", () => {
    const wasPlaying = playing;
    if (wasPlaying) pause();
    speed = Number(speedInput.value) / 100;
    speedOut.textContent = speedInput.value + "%";
    if (wasPlaying) play(position);
  });
  document.getElementById("print").addEventListener("click", () => window.print());
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !/^(BUTTON|INPUT)$/.test(e.target.tagName)) { e.preventDefault(); playing ? pause() : play(position); }
  });
  document.querySelector("main").addEventListener("click", (e) => {
    const g = e.target.closest("g.note");
    if (!g) return;
    const id = g.id.replace(/^c-/, "");
    const hit = notes.find((n) => n.id === id);
    if (!hit) return;
    if (playing) pause();
    clearLit();
    play(hit.start);
  });
  paint();
})();
