/**
 * Cloudflare Worker — the only server-side piece of the portfolio chat agent.
 *
 * The browser widget does retrieval client-side and sends the top chunks
 * here; this worker holds the Anthropic API key (a wrangler secret, never in
 * the repo), composes the role-conditioned prompt, calls the LLM, and logs
 * every request/response. Role prompts are fetched from the site's own
 * data/roles.json, so the client only ever sends a role id — prompt text
 * cannot be injected through the API surface.
 *
 * Routes:
 *   POST /chat  {session, role, question, history?, contexts[]} -> {answer, model}
 *   POST /log   client-side event record -> 204 (logging only)
 *   GET  /      -> health/info
 */

const LIMITS = {
  question: 1000, // chars
  contexts: 6,
  contextText: 1600,
  historyTurns: 8,
  historyText: 1200,
  logBytes: 4096,
};

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const cors = corsHeaders(origin, env);

    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
    if (!allowedOrigin(origin, env)) return json({ error: "origin not allowed" }, 403, cors);

    const url = new URL(request.url);
    try {
      if (request.method === "POST" && url.pathname === "/chat") return await handleChat(request, env, cors);
      if (request.method === "POST" && url.pathname === "/log") return await handleLog(request, env, cors);
      if (request.method === "GET" && url.pathname === "/") {
        return json({ service: "portfolio-chat", ok: true }, 200, cors);
      }
      return json({ error: "not found" }, 404, cors);
    } catch (err) {
      console.log(JSON.stringify({ type: "error", message: String(err && err.stack || err) }));
      return json({ error: "internal error" }, 500, cors);
    }
  },
};

function allowedOrigin(origin, env) {
  return (env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).includes(origin);
}

function corsHeaders(origin, env) {
  return {
    "Access-Control-Allow-Origin": allowedOrigin(origin, env) ? origin : "null",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function json(body, status, cors) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...cors },
  });
}

async function rateLimited(request, env) {
  if (!env.CHAT_KV) return false; // no KV bound -> rely on CORS + max_tokens caps
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  const hour = Math.floor(Date.now() / 3600_000);
  const key = `rl:${ip}:${hour}`;
  const count = parseInt((await env.CHAT_KV.get(key)) || "0", 10) + 1;
  await env.CHAT_KV.put(key, String(count), { expirationTtl: 3700 });
  return count > parseInt(env.RATE_LIMIT_PER_HOUR || "30", 10);
}

async function loadRoles(env) {
  const res = await fetch(`${env.SITE_BASE}/chat/data/roles.json`, {
    cf: { cacheTtl: 3600, cacheEverything: true },
  });
  if (!res.ok) throw new Error(`roles.json fetch failed: ${res.status}`);
  return res.json();
}

function validateChatBody(body) {
  if (!body || typeof body.question !== "string" || !body.question.trim()) return "question required";
  if (body.question.length > LIMITS.question) return "question too long";
  if (!Array.isArray(body.contexts) || body.contexts.length > LIMITS.contexts) return "bad contexts";
  for (const c of body.contexts) {
    if (typeof c.text !== "string" || c.text.length > LIMITS.contextText) return "bad context item";
  }
  if (body.history !== undefined) {
    if (!Array.isArray(body.history) || body.history.length > LIMITS.historyTurns) return "bad history";
    for (const h of body.history) {
      if (!["user", "assistant"].includes(h.role)) return "bad history role";
      if (typeof h.content !== "string" || h.content.length > LIMITS.historyText) return "bad history item";
    }
  }
  return null;
}

async function handleChat(request, env, cors) {
  if (!env.ANTHROPIC_API_KEY) {
    return json({ error: "worker not configured: missing ANTHROPIC_API_KEY secret" }, 503, cors);
  }
  if (await rateLimited(request, env)) return json({ error: "rate limited, try later" }, 429, cors);

  const body = await request.json().catch(() => null);
  const invalid = validateChatBody(body);
  if (invalid) return json({ error: invalid }, 400, cors);

  // Server-side off-topic guard, mirroring the widget's client-side gate:
  // no retrieved context means the question isn't about YC (or the caller
  // bypassed the widget) — refuse without spending an LLM call.
  if (body.contexts.length === 0) {
    const answer =
      "I can only answer questions about YC and his work — his projects, " +
      "skills, education, and publications. Nothing on the site matches " +
      "that question, so try asking about one of those instead.";
    await logRecord(env, {
      type: "chat_refused",
      ts: new Date().toISOString(),
      session: String(body.session || "").slice(0, 64),
      role: body.role,
      question: body.question,
    });
    return json({ answer, refused: true }, 200, cors);
  }

  const rolesData = await loadRoles(env);
  const role = rolesData.roles[body.role] || rolesData.roles[rolesData.default_role];

  const contextBlock = body.contexts
    .map((c, i) => `<chunk index="${i + 1}" page="${c.title || ""}" url="${c.url || ""}">\n${c.text}\n</chunk>`)
    .join("\n");
  const system =
    `${rolesData.base_system_prompt}\n\n` +
    `Visitor role: ${role.label}. ${role.system_prompt}\n\n` +
    `Context retrieved from the site for this question:\n${contextBlock || "(no relevant chunks found)"}`;

  const messages = [...(body.history || []), { role: "user", content: body.question }];

  const apiRes = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: env.LLM_MODEL || "claude-3-5-haiku-latest",
      max_tokens: 512,
      system,
      messages,
    }),
  });

  if (!apiRes.ok) {
    const detail = await apiRes.text();
    console.log(JSON.stringify({ type: "llm_error", status: apiRes.status, detail: detail.slice(0, 500) }));
    return json({ error: "llm call failed" }, 502, cors);
  }

  const data = await apiRes.json();
  const answer = (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("\n");

  await logRecord(env, {
    type: "chat",
    ts: new Date().toISOString(),
    session: String(body.session || "").slice(0, 64),
    role: body.role,
    question: body.question,
    retrieved: body.contexts.map((c) => ({ id: c.id, title: c.title, url: c.url })),
    answer,
    model: data.model,
    usage: data.usage,
  });

  return json({ answer, model: data.model }, 200, cors);
}

async function handleLog(request, env, cors) {
  const raw = await request.text();
  if (raw.length > LIMITS.logBytes) return json({ error: "log too large" }, 413, cors);
  let record;
  try {
    record = JSON.parse(raw);
  } catch {
    return json({ error: "bad json" }, 400, cors);
  }
  await logRecord(env, { type: "client_log", ts: new Date().toISOString(), ...record });
  return new Response(null, { status: 204, headers: cors });
}

async function logRecord(env, record) {
  // Always visible via `npx wrangler tail`; persisted 30 days when KV is bound.
  console.log(JSON.stringify(record));
  if (env.CHAT_KV) {
    const key = `log:${record.ts}:${crypto.randomUUID().slice(0, 8)}`;
    await env.CHAT_KV.put(key, JSON.stringify(record), { expirationTtl: 30 * 24 * 3600 });
  }
}
