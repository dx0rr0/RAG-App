const state = { channels: [], selectedChannel: null, selectedVideos: new Set(), conversationId: localStorage.getItem("ragConversationId"), messages: [] };
const $ = (selector) => document.querySelector(selector);

function toast(message, error = false) {
  const node = $("#toast"); node.textContent = message; node.classList.toggle("error", error); node.classList.add("visible");
  window.clearTimeout(toast.timer); toast.timer = window.setTimeout(() => node.classList.remove("visible"), 4200);
}

async function request(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Error HTTP ${response.status}`);
  return payload;
}

function currentVideos() {
  const channel = state.channels.find(item => item.id === state.selectedChannel);
  return channel ? channel.videos || [] : state.channels.flatMap(item => item.videos || []);
}

function renderChannels() {
  const root = $("#channels"); root.replaceChildren();
  if (!state.channels.length) { root.innerHTML = '<p class="muted">Todavía no tienes canales. Añade una URL arriba.</p>'; }
  state.channels.forEach(channel => {
    const button = document.createElement("button"); button.className = `channel-item${state.selectedChannel === channel.id ? " active" : ""}`;
    button.innerHTML = `<span class="channel-avatar">${escapeHtml(channel.title.slice(0, 1).toUpperCase())}</span><span class="channel-meta"><strong>${escapeHtml(channel.title)}</strong><small>${channel.video_count ?? channel.videos.length} vídeos</small></span><span class="channel-actions"><span class="arrow">›</span></span>`;
    button.addEventListener("click", () => { state.selectedChannel = channel.id; renderChannels(); renderVideos(); });
    button.addEventListener("contextmenu", event => { event.preventDefault(); refreshChannel(channel.id); }); root.append(button);
  });
  const select = $("#chat-scope"); const old = select.value; select.replaceChildren(new Option("Todos los canales", "all"));
  state.channels.forEach(channel => select.add(new Option(channel.title, channel.id)));
  if ([...select.options].some(item => item.value === old)) select.value = old;
}

function renderVideos() {
  const videos = currentVideos(); const root = $("#videos"); root.replaceChildren();
  if (!videos.length) root.innerHTML = '<p class="muted">No hay vídeos en este canal.</p>';
  videos.forEach(video => {
    const label = document.createElement("label"); label.className = "video-item";
    const box = document.createElement("input"); box.type = "checkbox"; box.checked = state.selectedVideos.has(video.id); box.disabled = video.status === "transcribed" || video.status === "queued" || video.status === "processing";
    box.addEventListener("change", () => { box.checked ? state.selectedVideos.add(video.id) : state.selectedVideos.delete(video.id); updateSelection(); });
    const status = video.status === "transcribed" ? "Listo" : video.status === "queued" || video.status === "processing" ? "En cola" : video.status === "failed" ? "Error" : "Pendiente";
    const duration = formatDuration(video.duration_seconds);
    label.append(box); const body = document.createElement("span"); body.className = "video-meta";
    body.innerHTML = `<strong>${escapeHtml(video.title)}</strong><small>${duration} · ${status}</small>`; label.append(body); root.append(label);
  });
  updateSelection();
}

function renderJobs(jobs = []) {
  const root = $("#jobs"); root.replaceChildren(); $("#job-count").textContent = jobs.filter(job => ["pending", "running"].includes(job.status)).length;
  if (!jobs.length) root.innerHTML = '<p class="muted">Sin trabajos todavía.</p>';
  jobs.slice(0, 8).forEach(job => {
    const row = document.createElement("div"); row.className = "job-item";
    const status = { pending: "En cola", running: "Procesando", completed: "Completado", failed: "Falló" }[job.status] || job.status;
    row.innerHTML = `<span class="status-dot ${escapeHtml(job.status)}"></span><span><strong>${escapeHtml(job.video_title)}</strong><small>${status} · estimado $${Number(job.estimated_cost_usd).toFixed(4)}</small></span>`; root.append(row);
  });
}

function renderMessages() {
  const root = $("#messages"); root.replaceChildren();
  if (!state.messages.length) { root.innerHTML = '<div class="empty-state"><div class="play-mark">▶</div><h2>Tu biblioteca de vídeos, consultable</h2><p>Añade un canal, selecciona los vídeos que quieres transcribir y aprueba el lote.</p></div>'; return; }
  state.messages.forEach(message => {
    const bubble = document.createElement("article"); bubble.className = `message ${message.role}`;
    const label = document.createElement("span"); label.className = "message-label"; label.textContent = message.role === "user" ? "TÚ" : "VÍDEOS RAG"; bubble.append(label);
    const text = document.createElement("div"); text.className = "message-text"; text.textContent = message.content; bubble.append(text);
    if (message.sources?.length) {
      const list = document.createElement("div"); list.className = "sources";
      message.sources.forEach(source => { const link = document.createElement("a"); link.href = source.url + (source.timestamp_seconds != null ? `&t=${source.timestamp_seconds}s` : ""); link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = `[${source.id}] ${source.title}${source.timestamp_seconds != null ? ` · ${formatDuration(source.timestamp_seconds)}` : ""}`; list.append(link); });
      bubble.append(list);
    }
    if (message.cost_usd != null) { const cost = document.createElement("small"); cost.className = "cost-label"; cost.textContent = `${message.cost_is_estimate ? "Coste estimado" : "Coste informado"}: $${Number(message.cost_usd).toFixed(6)}${message.verified ? " · respaldo verificado" : ""}`; bubble.append(cost); }
    else if (message.cost_unavailable) { const cost = document.createElement("small"); cost.className = "cost-label"; cost.textContent = `OpenRouter no informó el coste${message.verified ? " · respaldo verificado" : ""}`; bubble.append(cost); }
    root.append(bubble);
  });
  root.scrollTop = root.scrollHeight;
}

function updateSelection() {
  const videos = currentVideos(); const selected = videos.filter(video => state.selectedVideos.has(video.id));
  const estimate = selected.reduce((sum, video) => sum + Number(video.duration_seconds || 0) * 0.000003, 0);
  $("#selection-count").textContent = `${selected.length} vídeo${selected.length === 1 ? "" : "s"}`;
  $("#selection-estimate").textContent = `Estimación ASR: $${estimate.toFixed(4)}`;
  $("#approve-videos").disabled = !selected.length;
}

async function loadState() {
  const payload = await request("/api/state"); state.channels = payload.channels;
  if (state.selectedChannel && !state.channels.some(channel => channel.id === state.selectedChannel)) state.selectedChannel = null;
  renderChannels(); renderVideos(); renderJobs(payload.jobs);
}

async function refreshChannel(id) {
  try { await request(`/api/channels/${encodeURIComponent(id)}/refresh`, { method: "POST" }); await loadState(); toast("Canal actualizado. Los vídeos nuevos quedan pendientes de aprobación."); }
  catch (error) { toast(error.message, true); }
}

$("#channel-form").addEventListener("submit", async event => {
  event.preventDefault(); const button = event.submitter; button.disabled = true; button.textContent = "Leyendo…";
  try { const result = await request("/api/channels", { method: "POST", body: JSON.stringify({ url: $("#channel-url").value }) }); state.selectedChannel = result.channel.id; $("#channel-url").value = ""; await loadState(); toast("Canal añadido. Elige los vídeos y aprueba la transcripción."); }
  catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = "Añadir"; }
});

$("#refresh-all").addEventListener("click", async () => { for (const channel of state.channels) await refreshChannel(channel.id); });
$("#select-visible").addEventListener("click", () => { currentVideos().filter(video => !["transcribed", "queued", "processing"].includes(video.status)).forEach(video => state.selectedVideos.add(video.id)); renderVideos(); });
$("#approve-videos").addEventListener("click", async event => {
  const selected = currentVideos().filter(video => state.selectedVideos.has(video.id)); const cost = selected.reduce((sum, video) => sum + Number(video.duration_seconds || 0) * 0.000003, 0);
  if (!window.confirm(`¿Aprobar ${selected.length} vídeos? Estimación de transcripción: $${cost.toFixed(4)}. El lote se detiene si supera el límite configurado.`)) return;
  event.currentTarget.disabled = true;
  try { const result = await request("/api/ingestion/approve", { method: "POST", body: JSON.stringify({ video_ids: selected.map(video => video.id) }) }); state.selectedVideos.clear(); await loadState(); toast(`${result.queued} trabajos en cola. Estimación: $${Number(result.estimated_cost_usd).toFixed(4)}.`); }
  catch (error) { toast(error.message, true); }
  finally { updateSelection(); }
});

$("#chat-form").addEventListener("submit", async event => {
  event.preventDefault(); const input = $("#question"); const question = input.value.trim(); if (!question) return;
  const send = event.submitter; send.disabled = true; state.messages.push({ role: "user", content: question }); renderMessages(); input.value = "";
  const scope = $("#chat-scope").value;
  try {
    const result = await request("/api/chat", { method: "POST", body: JSON.stringify({ question, conversation_id: state.conversationId, channel_ids: scope === "all" ? [] : [scope], use_jev: $("#use-jev").checked, verify_faithfulness: $("#verify-faithfulness").checked }) });
    state.conversationId = result.conversation_id; localStorage.setItem("ragConversationId", result.conversation_id);
    state.messages.push({ role: "assistant", content: result.answer, sources: result.sources, cost_usd: result.cost_usd, cost_is_estimate: result.cost_is_estimate, cost_unavailable: result.cost_unavailable, verified: result.verified }); renderMessages();
  } catch (error) { state.messages.push({ role: "assistant", content: `No se pudo consultar: ${error.message}` }); renderMessages(); }
  finally { send.disabled = false; input.focus(); }
});

$("#new-chat").addEventListener("click", () => { state.conversationId = null; state.messages = []; localStorage.removeItem("ragConversationId"); renderMessages(); });
$("#question").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#chat-form").requestSubmit(); } });

function formatDuration(seconds) { const n = Number(seconds || 0); return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, "0")}`; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]); }

loadState().then(async () => {
  if (state.conversationId) {
    try {
      const saved = await request(`/api/conversations/${encodeURIComponent(state.conversationId)}`);
      state.messages = saved.messages.map(message => ({ role: message.role, content: message.content,
        sources: message.scope?.sources || [], verified: message.scope?.verified || false,
        cost_is_estimate: message.scope?.cost_is_estimate || false,
        cost_unavailable: message.scope?.cost_unavailable || false, cost_usd: message.cost_usd }));
      renderMessages();
    } catch (_) { state.conversationId = null; localStorage.removeItem("ragConversationId"); }
  }
}).catch(error => toast(error.message, true));
window.setInterval(() => request("/api/jobs").then(payload => { renderJobs(payload.jobs); return loadState(); }).catch(() => {}), 5000);
