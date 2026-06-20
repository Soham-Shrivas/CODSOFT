# CODSOFT

A collection of AI/ML projects. Each project is self-contained in its own directory.

---

## Projects

### 1. [Face Detection and Recognition](./Face%20detection%20and%20recognition)

Real-time face detection and recognition application with hand gesture support.

**Features:**
- **Face Detection** — Haar cascade (fast) and DNN SSD ResNet (accurate)
- **Face Recognition** — LBPH, EigenFace, and FisherFace algorithms
- **Finger Counting** — Real-time hand gesture detection with skin masking (HSV + YCrCb)
- **Modes** — Webcam live feed, image processing, video processing, training

**Tech:** Python, OpenCV, NumPy

```bash
pip install -r requirements.txt
python main.py --mode webcam --detect dnn --recognize --gesture
```

---

### 2. [TIC TAC TOE AI](./TIC%20TAC%20TOE%20AI)

Single-page HTML game with an unbeatable AI opponent.

**Features:**
- **Minimax + Alpha-Beta Pruning** — Perfect play on Hard difficulty
- **3 Difficulty Levels** — Easy (random), Medium (depth-limited), Unbeatable (full search)
- **Search Stats Panel** — Live node count, pruning stats, depth, think time
- **Animated SVG Marks** — X and O drawn with CSS animation
- **Responsive Design** — Works on desktop and mobile

**Tech:** HTML, CSS, JavaScript (vanilla, no dependencies)

```bash
# Just open index.html in a browser
open index.html
```

---

### 3. [CHATBOT WITH RULE-BASED RESPONSES](./CHATBOT%20WITH%20RULE-BASED%20RESPONSES)

A dual-mode chatbot with rule-based fallback and optional Groq AI backend.

**Features:**
- **AI Mode** — Connects to Groq's LLM API (Llama, Mixtral, Gemma models)
- **Rule-Based Fallback** — Pattern-matching responses when backend is offline
- **Settings Panel** — Model selection, API connection status
- **Quick Action Chips** — One-tap inquiry, support, and settings
- **Accessible** — ARIA labels, live regions, keyboard navigation

**Tech:** Node.js, Express, HTML/CSS (Tailwind), Groq API

```bash
cp .env.example .env   # Add your GROQ_API_KEY
npm install && npm start
```

---

## Project Structure

```
CODSOFT/
├── CHATBOT WITH RULE-BASED RESPONSES/
│   ├── index.html
│   ├── server.js
│   ├── package.json
│   └── .env.example
├── Face detection and recognition/
│   ├── main.py
│   ├── detector.py
│   ├── recognizer.py
│   ├── gesture.py
│   ├── utils.py
│   ├── config.py
│   ├── requirements.txt
│   └── data/
├── TIC TAC TOE AI/
│   └── index.html
└── README.md
```
