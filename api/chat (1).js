// api/chat.js
// This file runs only on Vercel's server, never in the visitor's browser.
// It reads the OpenAI API key from an environment variable, so the key is
// never written in the code and never sent to the website's visitors.

const MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const OPENAI_URL = "https://api.openai.com/v1/chat/completions";

// Limits that keep costs predictable.
const TEMP_MIN = 0;
const TEMP_MAX = 1.5; // OpenAI allows up to 2, but output often turns incoherent above ~1.5
const MAX_OUTPUT_TOKENS = 900; // cap on the length of each answer
const MAX_MESSAGES = 20; // only the most recent 20 messages are sent as context
const MAX_USER_CHARS = 1000; // max length of one visitor message
const MAX_ASSISTANT_CHARS = 6000; // max length of one earlier assistant reply
const COMPARE_RUNS = 3; // each temperature is run 3 times in a comparison
const TIMEOUT_MS = 45000;

const PROFILE_KEYS = ["budget", "use", "style", "materials", "wrist", "movement", "condition"];

const SYSTEM_PROMPT = `You are WristMatch, an AI watch concierge on a university project website. You help people choose a watch that suits their personality, budget and lifestyle, from complete beginners to enthusiasts.

Tone: warm, knowledgeable and concise, like a good boutique adviser who never pressures anyone. Use plain language and briefly explain jargon (for example "automatic", "complication", "lug-to-lug") the first time it appears. Reply in the language the user writes in.

Learning about the user:
- Over the conversation, find out: budget and currency; when they will wear the watch (everyday, work, special occasions, sport); personal style and preferred colours or materials; wrist size if known; preference for quartz, mechanical (hand-wound) or automatic, explaining the difference when useful; interest in new or pre-owned.
- Ask at most one or two questions per reply. Never ask again for something already known from the conversation or from "Known preferences" below.
- If the user asks a direct watch question, answer it directly. They do not have to complete an interview.
- Once you know the budget, the main use and at least one more preference, recommend. Do not keep interviewing endlessly.

Recommendations:
- Recommend about three real watches from established brands. Format each as a heading line "### 1. Brand Model line" followed by short "-" bullets in this order: "Why it fits you:", "Estimate:" (a rough price range, e.g. "Estimate: roughly €300–400 new"), "Strengths:", "Trade-offs:".
- Name each watch by brand and model line only, for example "Tissot PRX Powermatic 80" or "Seiko 5 Sports". Never give reference numbers, and never claim special, limited or colour editions.
- Finish with "**My first choice:**" and one or two sentences explaining why.
- Respect the stated budget. If nothing good fits, say so honestly and suggest how to adjust (for example pre-owned or a different movement type).
- Use "###" headings only for recommended watches, nowhere else.

Accuracy rules (these apply at every temperature setting):
- Only name a watch if you are confident it exists under that exact name. If unsure, choose a different, well-known model.
- Never invent specifications or prices. If unsure of a detail (case size, power reserve, current price), say so or leave it out.
- Prices are rough estimates from your training data. They may be outdated and vary by country, retailer and condition. Never claim current stock or availability, and never provide links.
- Never promise or imply that a watch will hold or increase its value. You may say that some models have historically held value well, while making clear that this is not guaranteed.
- When pre-owned watches come up, mention checking authenticity, box and papers, and the seller's reputation.

Scope: only discuss watches, wearing and styling them, owning and caring for them, and buying decisions. If asked about something unrelated, politely say you focus on watches and steer back with a watch-related question.

Formatting: short paragraphs, "-" bullets where helpful, **bold** sparingly, no tables, no code. Keep replies under about 250 words unless you are giving recommendations.`;

const PROFILE_PROMPT = `You keep a short summary of a user's watch preferences for a watch-advice website. Read the current summary and the conversation, then return the updated summary.
Rules:
- Use only what the USER stated or clearly confirmed. Never record something the assistant merely suggested.
- Keep values from the current summary unless the user changed them.
- Each value is a short phrase of at most 8 words, or null if unknown.
- budget includes the currency (e.g. "Up to €500"). use = when they'll wear it. materials = preferred colours or materials. wrist = wrist size. movement = quartz, automatic or hand-wound preference. condition = new or pre-owned.`;

const PROFILE_SCHEMA = {
  type: "object",
  properties: Object.fromEntries(PROFILE_KEYS.map((k) => [k, { type: ["string", "null"] }])),
  required: PROFILE_KEYS,
  additionalProperties: false,
};

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function cleanProfile(input) {
  const out = {};
  for (const key of PROFILE_KEYS) {
    const value = input && typeof input === "object" ? input[key] : null;
    out[key] = typeof value === "string" && value.trim() ? value.trim().slice(0, 80) : null;
  }
  return out;
}

function buildSystemPrompt(profile) {
  const known = PROFILE_KEYS.filter((k) => profile[k])
    .map((k) => `- ${k}: ${profile[k]}`)
    .join("\n");
  return `${SYSTEM_PROMPT}\n\nKnown preferences so far:\n${known || "- none yet"}`;
}

function isValidTemp(t) {
  return typeof t === "number" && Number.isFinite(t) && t >= TEMP_MIN && t <= TEMP_MAX;
}

function validateBody(body) {
  if (!body || typeof body !== "object") {
    return { error: "The request was empty or not valid JSON." };
  }
  const mode = body.mode;
  if (mode !== "chat" && mode !== "compare") {
    return { error: "Unknown request type." };
  }

  const { messages } = body;
  if (!Array.isArray(messages) || messages.length === 0) {
    return { error: "No messages were sent." };
  }
  const cleaned = [];
  for (const m of messages.slice(-MAX_MESSAGES)) {
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string") {
      return { error: "A message had an invalid format." };
    }
    const text = m.content.trim();
    if (!text) return { error: "Messages can't be empty." };
    if (m.role === "user" && text.length > MAX_USER_CHARS) {
      return { error: `Messages are limited to ${MAX_USER_CHARS} characters.` };
    }
    if (m.role === "assistant" && text.length > MAX_ASSISTANT_CHARS) {
      return { error: "The conversation is too long. Start a new conversation." };
    }
    cleaned.push({ role: m.role, content: text });
  }
  if (cleaned[cleaned.length - 1].role !== "user") {
    return { error: "The last message must come from the user." };
  }

  const result = { mode, messages: cleaned, profile: cleanProfile(body.profile) };
  const rangeError = `Temperature must be a number between ${TEMP_MIN} and ${TEMP_MAX}.`;

  if (mode === "chat") {
    if (!isValidTemp(body.temperature)) return { error: rangeError };
    result.temperature = Math.round(body.temperature * 10) / 10;
  } else {
    const temps = body.temperatures;
    if (!Array.isArray(temps) || temps.length !== 2 || !temps.every(isValidTemp)) {
      return { error: `Send two temperatures between ${TEMP_MIN} and ${TEMP_MAX} to compare.` };
    }
    result.temperatures = temps.map((t) => Math.round(t * 10) / 10);
    if (result.temperatures[0] === result.temperatures[1]) {
      return { error: "Choose two different temperatures to compare." };
    }
  }
  return result;
}

function mapOpenAIError(status, data) {
  const code = data && data.error ? data.error.code : null;
  const detail = data && data.error && data.error.message ? String(data.error.message).slice(0, 300) : "";
  if (status === 401) {
    return [502, "OpenAI rejected the API key. Check OPENAI_API_KEY in Vercel's environment variables, then redeploy."];
  }
  if (status === 403) {
    return [502, `This API key isn't allowed to use the model "${MODEL}". Ask your professor which models the key can use.`];
  }
  if (status === 404) {
    return [502, `The model "${MODEL}" isn't available for this API key.`];
  }
  if (status === 429) {
    if (code === "insufficient_quota") {
      return [429, "The API key has run out of credit. Your professor needs to top up the OpenAI account."];
    }
    return [429, "Too many requests at once. Wait about a minute, then try again."];
  }
  if (status === 400) {
    return [400, `OpenAI couldn't process the request${detail ? `: ${detail}` : "."}`];
  }
  if (status >= 500) {
    return [502, "OpenAI is having temporary problems. Try again in a moment."];
  }
  return [502, `Unexpected response from OpenAI (status ${status}).`];
}

async function postToOpenAI(apiKey, payload) {
  // Newer GPT-5.x models accept temperature only with reasoning switched off.
  if (/^gpt-5\.\d/.test(MODEL)) payload.reasoning_effort = "none";

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response;
  try {
    response = await fetch(OPENAI_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new ApiError(504, "OpenAI took too long to answer. Try again.");
    }
    throw new ApiError(502, "The server couldn't reach OpenAI. Try again in a moment.");
  } finally {
    clearTimeout(timer);
  }

  let data = null;
  try {
    data = await response.json();
  } catch (e) {
    data = null;
  }
  if (!response.ok) {
    const [status, message] = mapOpenAIError(response.status, data);
    throw new ApiError(status, message);
  }
  const choice = data && data.choices && data.choices[0];
  const content = choice && choice.message ? choice.message.content : null;
  if (typeof content !== "string" || !content.trim()) {
    throw new ApiError(502, "OpenAI returned an empty answer. Try again.");
  }
  return { content: content.trim(), finishReason: choice.finish_reason };
}

// The visible answer. Uses the temperature chosen on the website.
async function chatReply(apiKey, messages, temperature, profile) {
  const { content, finishReason } = await postToOpenAI(apiKey, {
    model: MODEL,
    messages: [{ role: "system", content: buildSystemPrompt(profile) }, ...messages],
    temperature,
    max_completion_tokens: MAX_OUTPUT_TOKENS,
  });
  if (finishReason === "length") {
    return `${content}\n\n*This answer reached the length limit. Ask me to continue.*`;
  }
  return content;
}

// The preference panel. A separate request, always at temperature 0, with a
// fixed JSON format, so it doesn't influence the temperature experiment.
async function extractProfile(apiKey, messages, profile) {
  const transcript = messages
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content.slice(0, 1500)}`)
    .join("\n\n");
  const { content } = await postToOpenAI(apiKey, {
    model: MODEL,
    temperature: 0,
    max_completion_tokens: 200,
    response_format: {
      type: "json_schema",
      json_schema: { name: "watch_preferences", strict: true, schema: PROFILE_SCHEMA },
    },
    messages: [
      { role: "system", content: PROFILE_PROMPT },
      { role: "user", content: `Current summary:\n${JSON.stringify(profile)}\n\nConversation:\n${transcript}` },
    ],
  });
  return cleanProfile(JSON.parse(content));
}

module.exports = async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "This address only accepts chat messages sent by the website." });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({
      error: "The server has no API key yet. Add OPENAI_API_KEY in Vercel under Settings, Environment Variables, then redeploy.",
    });
  }

  let body;
  try {
    body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
  } catch (e) {
    body = null;
  }
  const valid = validateBody(body);
  if (valid.error) return res.status(400).json({ error: valid.error });

  try {
    if (valid.mode === "chat") {
      const [reply, profile] = await Promise.all([
        chatReply(apiKey, valid.messages, valid.temperature, valid.profile),
        // If the panel update fails, the chat still works.
        extractProfile(apiKey, valid.messages, valid.profile).catch((err) => {
          console.error("Preference update failed:", err.message);
          return null;
        }),
      ]);
      return res.status(200).json({ model: MODEL, temperature: valid.temperature, reply, profile });
    }

    // Comparison: identical model, instructions, conversation and limits.
    // Only the temperature differs. Each temperature runs COMPARE_RUNS times.
    const results = await Promise.all(
      valid.temperatures.map(async (temperature) => {
        const replies = await Promise.all(
          Array.from({ length: COMPARE_RUNS }, () =>
            chatReply(apiKey, valid.messages, temperature, valid.profile)
          )
        );
        return { temperature, replies };
      })
    );
    return res.status(200).json({ model: MODEL, runs: COMPARE_RUNS, results });
  } catch (err) {
    if (err instanceof ApiError) return res.status(err.status).json({ error: err.message });
    console.error(err);
    return res.status(500).json({ error: "Something went wrong on the server. Try again." });
  }
};
