require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const API_KEY = process.env.GROQ_API_KEY;

if (!API_KEY) {
  console.warn('WARNING: GROQ_API_KEY not set in .env. AI features will be disabled.');
}

app.post('/api/chat', async (req, res) => {
  try {
    const { messages, model } = req.body;

    if (!API_KEY) {
      return res.status(503).json({
        error: 'API key not configured on server. Add GROQ_API_KEY to .env file.'
      });
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required.' });
    }

    const response = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: model || 'llama-3.1-8b-instant',
        messages: messages,
        temperature: 0.7,
        max_tokens: 1024
      })
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({
        error: data.error?.message || `Groq API error (${response.status})`
      });
    }

    res.json({ content: data.choices[0].message.content });
  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: err.message || 'Internal server error.' });
  }
});

app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    aiConfigured: !!API_KEY
  });
});

app.listen(PORT, () => {
  console.log(`Cognitive Assistant server running on http://localhost:${PORT}`);
  console.log(`AI mode: ${API_KEY ? 'enabled' : 'disabled (set GROQ_API_KEY in .env)'}`);
});
