const $ = (id) => document.getElementById(id);
let loadedSample = null; // remembers the dataset label of a loaded sample

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function setStep(step) {
  const steps = [...document.querySelectorAll("#pipeline li")];
  const idx = steps.findIndex((li) => li.dataset.step === step);
  steps.forEach((li, i) => {
    li.classList.toggle("done", i < idx || step === "all");
    li.classList.toggle("active", i === idx && step !== "all");
  });
}

async function loadSample(kind) {
  const r = await fetch(`/api/sample?kind=${kind}`);
  const d = await r.json();
  $("title").value = d.title;
  $("text").value = d.text;
  loadedSample = { kind: d.actual, title: d.title, text: d.text };
  $("sampleNote").textContent = `Loaded an unseen test article labelled ${d.actual}`;
}

function bars(listEl, words, cls) {
  listEl.innerHTML = "";
  if (!words.length) { listEl.innerHTML = '<li class="muted">none</li>'; return; }
  const max = Math.max(...words.map((w) => w[1]));
  for (const [w, v] of words.slice(0, 8)) {
    listEl.insertAdjacentHTML("beforeend",
      `<li><span class="w">${esc(w)}</span><span class="b ${cls}" style="width:${Math.max(6, (v / max) * 100)}%"></span></li>`);
  }
}

function list(el, items) {
  el.innerHTML = items.map((i) => `<li>${esc(i)}</li>`).join("");
}

async function analyze() {
  const title = $("title").value.trim();
  const text = $("text").value.trim();
  $("error").hidden = true;
  $("analyze").disabled = true;
  $("analyze").textContent = "Analyzing…";

  // Animate the pipeline while the server works
  const order = ["input", "pre", "tfidf", "ml", "nlp", "gen"];
  let i = 0;
  const timer = setInterval(() => setStep(order[Math.min(i++, order.length - 1)]), 250);

  try {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Something went wrong");
    render(d, title, text);
    setStep("all");
  } catch (e) {
    $("error").textContent = e.message;
    $("error").hidden = false;
    setStep("input");
  } finally {
    clearInterval(timer);
    $("analyze").disabled = false;
    $("analyze").textContent = "Analyze article →";
  }
}

function render(d, title, text) {
  const p = d.prediction;
  $("results").hidden = false;
  $("verdictCard").className = "card verdict " + (p.label === "FAKE" ? "is-fake" : "is-real");
  $("verdict").textContent = p.label;
  $("conf").textContent = p.confidence;
  $("pFake").textContent = p.prob_fake;
  $("pReal").textContent = p.prob_real;
  $("barFake").style.width = p.prob_fake + "%";
  $("barReal").style.width = p.prob_real + "%";
  $("modelName").textContent = p.model_name;
  $("elapsed").textContent = d.elapsed_ms;

  const unchanged = loadedSample && loadedSample.title === title && loadedSample.text === text;
  if (unchanged) {
    const ok = loadedSample.kind === p.label;
    $("actual").hidden = false;
    $("actual").className = "actual " + (ok ? "ok" : "bad");
    $("actual").textContent = `Actual label in dataset: ${loadedSample.kind} — ${ok ? "✓ correct prediction" : "✗ model was wrong"}`;
  } else {
    $("actual").hidden = true;
  }

  bars($("fakeWords"), p.fake_words, "fake");
  bars($("realWords"), p.real_words, "real");

  // Gen AI
  const g = d.genai;
  $("genSource").textContent = g.source;
  $("genSource").className = "tag " + (g.source.startsWith("Amazon") ? "aws" : "");
  $("genSummary").textContent = g.sections.summary;
  list($("genRed"), g.sections.red_flags);
  list($("genGood"), g.sections.credibility);
  list($("genVerify"), g.sections.verify);
  $("genDisclaimer").textContent = g.disclaimer;
  $("genPrompt").textContent = g.prompt;

  // Style signals
  const s = d.style;
  const tiles = [
    ["Words", s.word_count], ["Exclamation marks", s.exclamations], ["ALL-CAPS words", s.caps_words],
    ["Question marks", s.questions], ["Source attributions", s.attribution_count], ["Avg. sentence length", s.avg_sentence_len],
  ];
  $("stats").innerHTML = tiles.map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
  $("clickbait").innerHTML = s.clickbait_phrases.length
    ? `<p><b>Clickbait phrases:</b> ${s.clickbait_phrases.map((c) => `<span class="chip fake">${esc(c)}</span>`).join(" ")}</p>`
    : `<p class="muted">No clickbait phrases detected.</p>`;

  // NLP
  const n = d.nlp;
  $("nlpSource").textContent = n.source;
  $("nlpSource").className = "tag " + (n.source.startsWith("Amazon") ? "aws" : "");
  $("lang").textContent = n.language;
  $("sentiment").innerHTML = n.sentiment ? ` · <b>Sentiment:</b> ${esc(n.sentiment)}` : "";
  $("entities").innerHTML = n.entities.length
    ? n.entities.map((e) => `<span class="chip"><small>${esc(e.type)}</small> ${esc(e.text)}</span>`).join("")
    : '<span class="muted">No entities found.</span>';

  // Highlighted article
  $("highlighted").innerHTML = p.highlight
    .map(([t, tag]) => (tag ? `<mark class="hl-${tag}">${esc(t)}</mark>` : esc(t)))
    .join("");
  $("cleaned").textContent = p.cleaned_preview + (p.token_count > 60 ? " …" : "");

  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("analyze").addEventListener("click", analyze);
$("sampleFake").addEventListener("click", () => loadSample("FAKE"));
$("sampleReal").addEventListener("click", () => loadSample("REAL"));
$("clear").addEventListener("click", () => {
  $("title").value = ""; $("text").value = ""; loadedSample = null;
  $("sampleNote").textContent = ""; $("results").hidden = true; setStep("input");
});
setStep("input");
