const state = {
  sessionId: localStorage.getItem("altier_session_id") || null,
  busy: false,
};

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const sendEl = document.querySelector("#send");
const resetEl = document.querySelector("#reset");
const errorEl = document.querySelector("#error");
const contractEl = document.querySelector("#contract");
const cardsEl = document.querySelector("#cards");

const forbidden = [
  "[debug]",
  "ToolRequest",
  "ToolResult",
  "Contract(",
  "TravelContract(",
  "schema update",
  "chain of thought",
  "internal",
  "traceback",
  "Traceback",
  "route_semantics",
  "LLM prompt",
  "raw JSON",
  "stack trace",
  "Exception(",
];

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    inputEl.value = button.dataset.prompt;
    inputEl.focus();
  });
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text || state.busy) return;
  inputEl.value = "";
  await sendMessage(text);
});

resetEl.addEventListener("click", async () => {
  if (state.sessionId) {
    await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}`, { method: "DELETE" }).catch(() => {});
  }
  state.sessionId = null;
  localStorage.removeItem("altier_session_id");
  messagesEl.innerHTML = "";
  renderContract({});
  renderCards([]);
  clearError();
});

boot();

async function boot() {
  if (!state.sessionId) {
    await createSession();
    return;
  }
  const response = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}`);
  if (!response.ok) {
    state.sessionId = null;
    localStorage.removeItem("altier_session_id");
    await createSession();
    return;
  }
  const data = await response.json();
  renderMessages(data.messages || []);
  renderContract(data.contract_summary || {});
  renderCards(data.cards || []);
}

async function createSession() {
  const response = await fetch("/api/sessions", { method: "POST" });
  const data = await response.json();
  state.sessionId = data.session_id;
  localStorage.setItem("altier_session_id", state.sessionId);
  renderContract(data.contract_summary || {});
  renderCards([]);
}

async function sendMessage(text) {
  clearError();
  setBusy(true);
  appendMessage("user", text);
  const assistantEl = appendMessage("assistant", "");
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message: text, stream: true }),
    });
    if (!response.ok || !response.body) {
      throw new Error("Request failed.");
    }
    await readSse(response.body, {
      token: (data) => {
        if (!safeText(data.text || "")) return;
        assistantEl.textContent += data.text || "";
        messagesEl.scrollTop = messagesEl.scrollHeight;
      },
      final: (data) => {
        if (data.assistant_response && safeText(data.assistant_response)) {
          assistantEl.textContent = data.assistant_response;
        }
        renderContract(data.contract_summary || {});
        renderCards(data.cards || []);
      },
      error: (data) => {
        assistantEl.textContent = data.message || "Could not process request.";
        showError(assistantEl.textContent);
      },
    });
  } catch (_error) {
    assistantEl.textContent = "Could not process request. Please try again.";
    showError("Could not process request. No booking/payment action was taken.");
  } finally {
    setBusy(false);
  }
}

async function readSse(stream, handlers) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const event = parseSse(part);
      if (event && handlers[event.event]) handlers[event.event](event.data);
    }
  }
}

function parseSse(block) {
  const eventLine = block.split("\n").find((line) => line.startsWith("event:"));
  const dataLine = block.split("\n").find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return null;
  const event = eventLine.slice("event:".length).trim();
  try {
    return { event, data: JSON.parse(dataLine.slice("data:".length).trim()) };
  } catch (_error) {
    return null;
  }
}

function renderMessages(messages) {
  messagesEl.innerHTML = "";
  for (const message of messages) {
    if (safeText(message.content || "")) appendMessage(message.role, message.content || "");
  }
}

function appendMessage(role, content) {
  const el = document.createElement("div");
  el.className = `message ${role === "user" ? "user" : "assistant"}`;
  el.textContent = safeText(content) ? content : "Content hidden for safety.";
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function renderContract(summary) {
  if (!summary || Object.keys(summary).length === 0) {
    contractEl.textContent = "No active travel contract yet.";
    return;
  }
  contractEl.classList.remove("muted-text");
  contractEl.innerHTML = "";
  const fields = [
    ["Route", formatRoute(summary.route)],
    ["Dates", formatObject(summary.dates)],
    ["Budget", formatObject(summary.budget)],
    ["Companions", formatObject(summary.companions)],
    ["Preferences", formatObject(summary.preferences)],
    ["Missing fields", (summary.missing_fields || []).join(", ")],
    ["Warnings", (summary.warnings || []).join(" / ")],
  ];
  for (const [label, value] of fields) {
    if (!value) continue;
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = `<label></label><div></div>`;
    field.querySelector("label").textContent = label;
    field.querySelector("div").textContent = value;
    contractEl.appendChild(field);
  }
}

function renderCards(cards) {
  cardsEl.innerHTML = "";
  cardsEl.classList.remove("muted-text");
  if (!cards || cards.length === 0) {
    cardsEl.textContent = "Cards appear after planning turns.";
    cardsEl.classList.add("muted-text");
    return;
  }
  for (const card of cards) {
    if (!safeText(JSON.stringify(card))) continue;
    const el = document.createElement("article");
    el.className = "card";
    const title = document.createElement("h3");
    title.textContent = card.title || card.type || "Card";
    el.appendChild(title);
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = [card.classification, card.source, card.safety_label].filter(Boolean).join(" · ");
    if (meta.textContent) el.appendChild(meta);
    el.appendChild(renderCardBody(card));
    cardsEl.appendChild(el);
  }
}

function renderCardBody(card) {
  const wrapper = document.createElement("div");
  if (card.type === "itinerary") {
    for (const item of card.items || []) {
      const p = document.createElement("p");
      p.textContent = `Day ${item.day}: ${item.title}`;
      wrapper.appendChild(p);
      wrapper.appendChild(list([...(item.morning || []), ...(item.afternoon || []), ...(item.evening || [])].slice(0, 5)));
    }
    return wrapper;
  }
  if (card.type === "cost_estimate") {
    const total = card.total || {};
    const p = document.createElement("p");
    p.textContent = `Total rough range: ${total.min || "?"}–${total.max || "?"} ${total.currency || card.currency || "USD"}`;
    wrapper.appendChild(p);
  }
  const items = (card.items || []).map((item) => {
    if (typeof item === "string") return item;
    if (item.message) return `${item.level || "info"}: ${item.message}`;
    if (item.category) return `${item.category}: ${item.min || ""}${item.max ? "–" + item.max : ""} ${item.currency || ""} ${item.source_type || ""}`;
    if (item.route) return `${item.route} · $${item.price_usd} · ${item.safety_label || card.safety_label || "demo/mock"}`;
    return JSON.stringify(item);
  });
  wrapper.appendChild(list(items));
  return wrapper;
}

function list(items) {
  const ul = document.createElement("ul");
  for (const item of items.filter(Boolean)) {
    const li = document.createElement("li");
    li.textContent = item;
    ul.appendChild(li);
  }
  return ul;
}

function formatRoute(route = {}) {
  return [route.origin, route.destination].filter(Boolean).join(" → ");
}

function formatObject(value = {}) {
  if (!value || Object.keys(value).length === 0) return "";
  return Object.entries(value)
    .filter(([, v]) => v !== "" && v !== null && !(Array.isArray(v) && v.length === 0))
    .map(([k, v]) => `${k}: ${Array.isArray(v) ? JSON.stringify(v) : v}`)
    .join(" · ");
}

function safeText(text) {
  return !forbidden.some((marker) => text.toLowerCase().includes(marker.toLowerCase()));
}

function setBusy(value) {
  state.busy = value;
  sendEl.disabled = value;
  inputEl.disabled = value;
}

function showError(message) {
  errorEl.textContent = message;
}

function clearError() {
  errorEl.textContent = "";
}
