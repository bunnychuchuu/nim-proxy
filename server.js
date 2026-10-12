// server.js — Robust Hybrid OpenAI ↔ NIM Proxy
// Express 5 Compatible
// Fixes: auth bypass, startup DDoS, silent stream failures, memory leaks, Express 5 deprecations

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}

validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'yi-large': '01-ai/yi-large',
  'fuyu-8b': 'adept/fuyu-8b',
  'jamba-1.5-large-instruct': 'ai21labs/jamba-1.5-large-instruct',
  'sea-lion-7b-instruct': 'aisingapore/sea-lion-7b-instruct',
  'starcoder2-15b': 'bigcode/starcoder2-15b',
  'dbrx-instruct': 'databricks/dbrx-instruct',
  'deepseek-coder-6.7b-instruct': 'deepseek-ai/deepseek-coder-6.7b-instruct',
  'deepseek-v4.1-flash': 'deepseek-ai/deepseek-v4.1-flash',
  'codegemma-1.1-7b': 'google/codegemma-1.1-7b',
  'codegemma-7b': 'google/codegemma-7b',
  'deplot': 'google/deplot',
  'diffusiongemma-26b-a4b-it': 'google/diffusiongemma-26b-a4b-it',
  'gemma-2b': 'google/gemma-2b',
  'gemma-3-12b-it': 'google/gemma-3-12b-it',
  'gemma-3-4b-it': 'google/gemma-3-4b-it',
  'gemma-4-31b-it': 'google/gemma-4-31b-it',
  'recurrentgemma-2b': 'google/recurrentgemma-2b',
  'granite-3.0-3b-a800m-instruct': 'ibm/granite-3.0-3b-a800m-instruct',
  'granite-3.0-8b-instruct': 'ibm/granite-3.0-8b-instruct',
  'granite-34b-code-instruct': 'ibm/granite-34b-code-instruct',
  'granite-8b-code-instruct': 'ibm/granite-8b-code-instruct',
  'codellama-70b': 'meta/codellama-70b',
  'llama-3.2-11b-vision-instruct': 'meta/llama-3.2-11b-vision-instruct',
  'llama-3.2-90b-vision-instruct': 'meta/llama-3.2-90b-vision-instruct',
  'llama-guard-4-12b': 'meta/llama-guard-4-12b',
  'llama2-70b': 'meta/llama2-70b',
  'muse-glimmer-30b': 'meta/muse-glimmer-30b',
  'kosmos-2': 'microsoft/kosmos-2',
  'phi-3-vision-128k-instruct': 'microsoft/phi-3-vision-128k-instruct',
  'phi-3.5-moe-instruct': 'microsoft/phi-3.5-moe-instruct',
  'codestral-22b-instruct-v0.1': 'mistralai/codestral-22b-instruct-v0.1',
  'mistral-7b-instruct-v0.3': 'mistralai/mistral-7b-instruct-v0.3',
  'mistral-large': 'mistralai/mistral-large',
  'mistral-large-2-instruct': 'mistralai/mistral-large-2-instruct',
  'mixtral-8x22b-v0.1': 'mistralai/mixtral-8x22b-v0.1',
  'kimi-k2.6': 'moonshotai/kimi-k2.6',
  'kimi-k3': 'moonshotai/kimi-k3',
  'mistral-nemo-12b-instruct': 'nv-mistralai/mistral-nemo-12b-instruct',
  'ai-synthetic-video-detector': 'nvidia/ai-synthetic-video-detector',
  'cosmos-reason2-8b': 'nvidia/cosmos-reason2-8b',
  'embed-qa-4': 'nvidia/embed-qa-4',
  'ising-calibration-1.5-31b': 'nvidia/ising-calibration-1.5-31b',
  'llama-3.1-nemoguard-8b-content-safety': 'nvidia/llama-3.1-nemoguard-8b-content-safety',
  'llama-3.1-nemoguard-8b-topic-control': 'nvidia/llama-3.1-nemoguard-8b-topic-control',
  'llama-3.1-nemotron-51b-instruct': 'nvidia/llama-3.1-nemotron-51b-instruct',
  'llama-3.1-nemotron-70b-instruct': 'nvidia/llama-3.1-nemotron-70b-instruct',
  'llama-3.1-nemotron-safety-guard-8b-v3': 'nvidia/llama-3.1-nemotron-safety-guard-8b-v3',
  'llama-3.1-nemotron-ultra-253b-v1': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'llama-3.2-nemoretriever-1b-vlm-embed-v1': 'nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1',
  'llama-3.2-nv-embedqa-1b-v1': 'nvidia/llama-3.2-nv-embedqa-1b-v1',
  'llama-nemotron-embed-vl-1b-v2': 'nvidia/llama-nemotron-embed-vl-1b-v2',
  'llama3-chatqa-1.5-70b': 'nvidia/llama3-chatqa-1.5-70b',
  'mistral-nemo-minitron-8b-8k-instruct': 'nvidia/mistral-nemo-minitron-8b-8k-instruct',
  'nemotron-3-embed-1b': 'nvidia/nemotron-3-embed-1b',
  'nemotron-3-nano-omni-30b-a3b-reasoning': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
  'nvidia/nemotron-3-super-120b-a12b': 'nvidia/nemotron-3-super-120b-a12b',
  'nemotron-3-ultra-550b-a55b': 'nvidia/nemotron-3-ultra-550b-a55b',
  'nemotron-3.5-content-safety': 'nvidia/nemotron-3.5-content-safety',
  'nemotron-3.5-lightning-30b-a3b': 'nvidia/nemotron-3.5-lightning-30b-a3b',
  'nemotron-4-340b-instruct': 'nvidia/nemotron-4-340b-instruct',
  'nemotron-4-340b-reward': 'nvidia/nemotron-4-340b-reward',
  'nemotron-nano-3-30b-a3b': 'nvidia/nemotron-nano-3-30b-a3b',
  'nemotron-parse': 'nvidia/nemotron-parse',
  'nemotron-parse-2.0': 'nvidia/nemotron-parse-2.0',
  'neva-22b': 'nvidia/neva-22b',
  'nv-embedqa-mistral-7b-v2': 'nvidia/nv-embedqa-mistral-7b-v2',
  'nvclip': 'nvidia/nvclip',
  'riva-translate-4b-instruct': 'nvidia/riva-translate-4b-instruct',
  'riva-translate-4b-instruct-v2': 'nvidia/riva-translate-4b-instruct-v2',
  'vila': 'nvidia/vila',
  'gpt-oss-20b': 'openai/gpt-oss-20b',
  'laguna-xs-2.1': 'poolside/laguna-xs-2.1',
  'arctic-embed-l': 'snowflake/arctic-embed-l',
  'palmyra-creative-122b': 'writer/palmyra-creative-122b',
  'palmyra-fin-70b-32k': 'writer/palmyra-fin-70b-32k',
  'palmyra-med-70b': 'writer/palmyra-med-70b',
  'palmyra-med-70b-32k': 'writer/palmyra-med-70b-32k',
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-5.3-flash': 'z-ai/glm-5.3-flash',
  'zamba2-7b-instruct': 'zyphra/zamba2-7b-instruct'
};

const FALLBACK_MODELS = [
  'mistralai/mistral-medium-3.5-128b',
  'mistralai/mistral-small-4-119b-2603',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'google/gemma-4-31b-it'
];

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// FIX: Extract token AFTER "Bearer " prefix, compare only the token
// Prevents bypass when CLIENT_AUTH_KEY is empty (expected would be "Bearer " which is 7 chars)
function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);
  
  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────────

// FIX: Use lightweight model listing instead of burning inference quota
// If NIM doesn't support /models, skip validation entirely rather than DDoS-ing yourself
async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');

  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

    const invalid = [];
    
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }

  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
    console.warn('[VALIDATION] Consider setting SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.');
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000 });
    console.log('[DISCORD] Alert sent.');
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────

// FIX: Wrap res.write in try/catch to prevent crashes on closed sockets
function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Helper: Fallback Chain ─────────────────────────────────────────────────

async function callWithFallback(baseRequest, models) {
  let lastError = null;

  for (const model of models) {
    try {
      const res = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        { ...baseRequest, model },
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS
        }
      );

      return { response: res, model };

    } catch (err) {
      lastError = err;
      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        err.response?.status,
        err.response?.data?.error?.message || err.message
      );
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.1.0' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    const primaryModel = MODEL_MAPPING[model] || 'nvidia/llama-3.3-nemotron-super-49b-v1.5';
    const modelChain = [primaryModel, ...FALLBACK_MODELS];

    const baseRequest = {
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false,
      extra_body: ENABLE_THINKING_MODE
        ? { chat_template_kwargs: { thinking: true } }
        : undefined
    };

    const { response, model: usedModel } = await callWithFallback(baseRequest, modelChain);
    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        if (line.includes('[DONE]')) {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            let content = delta.content || '';
            const reasoning = delta.reasoning_content;

            if (SHOW_REASONING) {
              if (reasoning && !reasoningOpen) {
                content = `<thinking>\n${reasoning.replace(/\n/g, '\\n')}`;
                reasoningOpen = true;
              } else if (reasoning) {
                content = reasoning.replace(/\n/g, '\\n');
              }

              if (delta.content && reasoningOpen) {
                content += `\n</thinking>\n\n${delta.content}`;
                reasoningOpen = false;
              }
            }

            delta.content = content;
            delete delta.reasoning_content;
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);

        } catch (parseErr) {
          // FIX: Don't silently swallow—send error to client so they know data was lost
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { 
              message: 'Upstream sent malformed chunk', 
              type: 'stream_parse_error',
              details: line.slice(0, 100)
            } 
          })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { 
              message: 'Stream buffer overflow', 
              type: 'stream_error' 
            } 
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();

        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }

        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      // FIX: Check req.destroyed (Node/Express 5) 
      // Don't destroy already-finished streams
      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;
        
        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }

        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => {
          let content = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            const safeReasoning = choice.message.reasoning_content.replace(/\n/g, '\\n');
            content = `<thinking>\n${safeReasoning}\n</thinking>\n\n${content}`;
          }

          return {
            index: i,
            message: {
              role: choice.message?.role || 'assistant',
              content,
              tool_calls: choice.message?.tool_calls
            },
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    console.error('[PROXY] NIM response:', error.response?.data);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    // Clean up upstream stream if we have it
    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

// FIX: Express 5 named wildcard — but use proper 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
  
  // Run validation after server starts, non-blocking
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
  
