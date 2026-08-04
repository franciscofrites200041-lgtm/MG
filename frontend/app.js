// SPA minimal: form -> POST /api/search -> render de hits.
// Cita path + pagina + snippet. Click en hit -> POST /api/read_page para el texto completo.

const API = "/api";

const form = document.getElementById("search-form");
const qInput = document.getElementById("q");
const topKSel = document.getElementById("top_k");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = qInput.value.trim();
  if (!query) return;
  await runSearch(query, parseInt(topKSel.value, 10));
});

async function runSearch(query, top_k) {
  statusEl.textContent = "Buscando...";
  resultsEl.innerHTML = "";
  const t0 = performance.now();
  try {
    const res = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const data = await res.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    renderHits(data.hits || [], data.text, dt);
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
}

function renderHits(hits, fallbackText, dt) {
  if (!hits.length && fallbackText) {
    statusEl.textContent = `Respuesta en ${dt}s (backend SQLite)`;
    resultsEl.innerHTML = `<pre class="whitespace-pre-wrap bg-white p-4 rounded-lg border">${escapeHtml(fallbackText)}</pre>`;
    return;
  }
  if (!hits.length) {
    statusEl.textContent = `Sin resultados (${dt}s)`;
    return;
  }
  statusEl.textContent = `${hits.length} resultados en ${dt}s`;
  resultsEl.innerHTML = hits
    .map((h, i) => hitCard(h, i))
    .join("");
  document.querySelectorAll("[data-read-page]").forEach((btn) => {
    btn.addEventListener("click", () => readPage(btn.dataset.path, parseInt(btn.dataset.page, 10)));
  });
}

function hitCard(h, i) {
  const scorePct = h.score ? ` &middot; score ${(h.score * 100).toFixed(1)}%` : "";
  return `
    <article class="bg-white p-4 rounded-lg border border-slate-200 hover:border-slate-400 transition">
      <div class="flex items-baseline justify-between gap-3">
        <h3 class="font-medium text-slate-900">
          <span class="text-slate-400 text-sm mr-2">#${i + 1}</span>
          ${escapeHtml(h.filename || "?")}
          <span class="text-slate-500 text-sm font-normal">&middot; pag. ${h.page ?? "?"}${scorePct}</span>
        </h3>
        <button
          data-read-page
          data-path="${escapeHtml(h.path)}"
          data-page="${h.page}"
          class="text-sm text-slate-600 hover:text-slate-900 underline"
        >Leer pagina</button>
      </div>
      <p class="text-sm text-slate-500 mt-1">${escapeHtml(h.path)}</p>
      <p class="text-slate-700 mt-2">${escapeHtml((h.snippet || "").slice(0, 500))}</p>
    </article>
  `;
}

async function readPage(path, page) {
  statusEl.textContent = `Leyendo ${path} pag. ${page}...`;
  try {
    const res = await fetch(`${API}/read_page`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, page }),
    });
    const data = await res.json();
    resultsEl.innerHTML = `
      <div class="bg-white p-6 rounded-lg border">
        <div class="flex justify-between mb-4">
          <h2 class="font-medium">${escapeHtml(path)} &middot; pag. ${page}</h2>
          <button id="back" class="text-slate-600 hover:text-slate-900 underline">Volver</button>
        </div>
        <pre class="whitespace-pre-wrap text-slate-800">${escapeHtml(data.text || "")}</pre>
      </div>
    `;
    document.getElementById("back").addEventListener("click", () => form.dispatchEvent(new Event("submit")));
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = `Error leyendo pagina: ${err.message}`;
  }
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
