import os
import re
import uuid
import json
import time
import random
import traceback
import logging
from typing import Optional, List, Tuple

import requests
import wikipedia
import feedparser
import sympy as sp
import numpy as np
import io, base64
import cv2

from bs4 import BeautifulSoup
from langdetect import detect
from word2number import w2n
from datetime import datetime,timezone, timedelta
from googleapiclient.discovery import build
from models import User,ChatHistory,ChatSession
from extensions import db
from flask_login import login_user,login_required, current_user,logout_user,LoginManager,UserMixin
from flask_migrate import Migrate
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from flask import Flask, request, jsonify, render_template, send_from_directory, abort,url_for,redirect,flash
from werkzeug.security import generate_password_hash,check_password_hash

# GPT4All import — if not installed, we handle gracefully
try:
    from gpt4all import GPT4All
except Exception:
    GPT4All = None

# local helper (from your repo) — if missing, get best-effort behavior
try:
    from smart_topic_cleaner import preprocess_for_wikipedia
except Exception:
    def preprocess_for_wikipedia(q): return q

# -----------------------
# Config / Logging
# -----------------------
app = Flask(__name__)
app.secret_key = "Aarochatbot"
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"   # users.db file create aagum
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
migrate = Migrate(app, db)

with app.app_context():
    db.create_all()


# --- Flask-Login setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


load_dotenv()  # load .env first
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") 
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
GOOGLE_CX=os.getenv("GOOGLE_CX")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    app.logger.warning("YOUTUBE_API_KEY not set. YouTube searches will fail if requested.")

USERS_FILE = "users.json"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------
# Globals
# -----------------------
chat_histories = []
sessions = {}
current_session_id = None

# -----------------------
# Model load (non-blocking)
# -----------------------
model_path = os.path.expanduser("C:/Users/roshi/Downloads/ai_chatbot_with_math_backend/models/ggml-gpt4all-j.bin")
gpt = None
if GPT4All is not None:
    try:
        app.logger.info("Looking for GPT4All model at: %s", model_path)
        gpt = GPT4All(model_path, allow_download=False)
        # Many builds require .open() for some versions; handle if present
        if hasattr(gpt, "open"):
            try:
                gpt.open()
            except Exception:
                pass
        app.logger.info("GPT4All model initialized.")
    except Exception as e:
        app.logger.exception("Model load error (continuing without model): %s", e)
        gpt = None
else:
    app.logger.warning("gpt4all package not available. Text-generation disabled.")

BOT_NAME = "Aaro"

#----------------------
#Tanglish Function
#----------------------
# --- Tanglish reply function ---
def tanglish_reply(sess,user_message, max_tokens=200):
    """
    Generate Tanglish reply using GPT4All with proper chat history handling.
    """
    style_prompt = (
         "You are a fun chatbot who always replies in tamil or tamil+eng "
         "Respond casually, like a normal person talking"
         "(Tamil+English mix). Use casual words like 'naa', 'nee', 'da', "
         "with emojis 😄. "
          "always a mix. Example:\n\n"
          "User: Hi, epdi iruka?\n"
          "Bot: Naa super da 😎 Nee epdi iruka?\n\n"
          "User: Saptiya?\n"
          "Bot: Haa da, dosa saapten 😋 Nee?\n\n"
          "Now continue the chat below:\n"
    )

    # extract last 10 messages (5 user-bot pairs)
    context_lines = []
    for m in sess.get("messages", [])[-10:]:
        role = "User" if m["role"]=="user" else "Bot"
        context_lines.append(f"{role}: {m['content']}")

    context_text = "\n".join(context_lines)

    prompt = f"{style_prompt}{context_text}\nUser: {user_message}\nBot:"

    # generate reply using GPT4All
    response = gpt.generate(prompt, max_tokens=max_tokens)
    if isinstance(response, list):
        response = "".join(response).strip()

    # post-process emojis/slang
    response = response.replace("yes", "haa 😄").replace("no", "illa 😅")

    sess["messages"].append({"role":"user","content":user_message})
    sess["messages"].append({"role": "bot","content":response})

    
    return response.strip()

# -----------------------
# Users (learning agent)
# -----------------------

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(data):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        app.logger.exception("Failed to save users.json: %s", e)

def ensure_user_profile(user_id):
    users = load_users()
    if user_id not in users:
        users[user_id] = {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "preferred_language": None,
            "topics_count": {},
            "emotion_history": [],  # last N emotions
        }
        save_users(users)
    return users[user_id]

def update_user_topic(user_id, topic):
    users = load_users()
    ensure_user_profile(user_id)
    if topic:
        users = load_users()
        users[user_id]["topics_count"][topic] = users[user_id]["topics_count"].get(topic, 0) + 1
        save_users(users)

def update_user_emotion(user_id, emotion):
    users = load_users()
    ensure_user_profile(user_id)
    hist = users[user_id].get("emotion_history", [])
    hist.append({"emotion": emotion, "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    # keep last 10
    users[user_id]["emotion_history"] = hist[-10:]
    save_users(users)

def set_user_language(user_id, lang_code):
    users = load_users()
    ensure_user_profile(user_id)
    users[user_id]["preferred_language"] = lang_code
    save_users(users)

# -----------------------
# Emotion & keyword detection
# -----------------------
EMOTION_SYNONYMS_EN = {
    "happy": ["happy", "glad", "excited", "awesome", "great", "fantastic", "good mood"],
    "sad": ["sad", "down", "upset", "unhappy", "depressed", "low"],
    "angry": ["angry", "mad", "furious", "irritated", "annoyed"],
    "anxious": ["anxious", "nervous", "tense", "worried", "panic", "stressed"],
    "stressed": ["stressed", "overwhelmed", "burned out", "pressure", "too much work"],
    "lonely": ["lonely", "alone", "no one", "left out"],
    "bored": ["bored", "nothing to do", "meh"],
    "tired": ["tired", "exhausted", "sleepy", "fatigued", "drained"],
    "confused": ["confused", "don't get it", "not sure", "lost"],
    "grateful": ["grateful", "thankful", "blessed", "appreciative"],
    "proud": ["proud", "achieved", "accomplished", "did it"],
    "guilty": ["guilty", "regret", "ashamed", "sorry"],
    "scared": ["scared", "afraid", "fear", "terrified"],
    "hurt": ["hurt", "heartbroken", "broken", "pain"],
    "motivated": ["motivated", "inspired", "pumped", "driven"],
    "hopeful": ["hopeful", "optimistic", "positive"],
    "jealous": ["jealous", "envious"],
    "love": ["love", "loving", "affection"],
}
EMOTION_SYNONYMS = {}
for k in set(list(EMOTION_SYNONYMS_EN.keys())):
    EMOTION_SYNONYMS[k] = list(set(EMOTION_SYNONYMS_EN.get(k, [])))

EMOTION_REPLIES = {
    "happy": ["🎉 That’s awesome! I’m really happy for you!", "😄 Love that energy! What made your day so good?", "🙌 Keep that vibe going! Tell me more!"],
    "sad": ["😔 I’m here for you… want to talk about what’s weighing on you?", "💛 Sorry you’re feeling low. A small step today can still be a win.", "🤝 You’re not alone. Share a bit—sometimes that lightens the load."],
    "angry": ["😤 That sounds really frustrating. Want to vent it out safely here?", "🧘 Let’s take a slow breath together. What exactly triggered it?", "⚡ Your feelings are valid. We can think through next steps calmly."],
    "anxious": ["😟 That jittery feeling is tough. Try a 4-7-8 breath with me?", "🫶 You’re safe here. What’s the main worry looping in your mind?", "🌿 Let’s break it down into smaller pieces we can handle."],
    "stressed": ["🧠 Heavy load, huh? Let’s list the top 3 things and knock one out.", "🗓️ Tiny plan time: what’s the smallest next step you can do in 5 mins?", "💆 A pause helps. Hydrate, one deep breath, then we move."],
    "neutral": ["🙂 I’m listening.", "💬 Tell me more about that.", "👀 Interesting… go on."],
    "love": ["❤️ That’s so sweet of you!", "🥰 I appreciate your kindness.", "💖 Sending you good vibes!"]
}
CUSTOM_KEYWORDS = {
    "name": [f"My name’s {BOT_NAME}! 😊 Nice to meet you.", f"I’m {BOT_NAME}, your friendly AI buddy 🤖", f"You can call me {BOT_NAME} — I’m here to help! 😄"],
    "who are you": [f"I’m {BOT_NAME}, your chat companion. Ask me anything!", f"{BOT_NAME} here—math, notes, motivation, I’ve got you."],
    "joke": ["Why did the math book look sad? It had too many problems. 😆", "Parallel lines have so much in common. It’s a shame they’ll never meet. 😂"],
    "motivate": ["You don’t need to be perfect—just consistent. Tiny steps win. 💪", "Your future self is cheering. Do one small thing right now. 🚀"],
}
BOT_NAME_PATTERNS = [
    r"\bwhat(?:'s| is)?\s+(?:your|ur)\s+name\??\b",
    r"\btell\s+me\s+your\s+name\??\b",
    r"\byour\s+name\??\b",
]

def _match_any(patterns: List[str], text: str) -> bool:
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False

def detect_intent(user_text: str) -> Tuple[Optional[str], float]:
    txt = user_text.lower()
    # name patterns
    if _match_any(BOT_NAME_PATTERNS, txt):
        return "name", 0.99
    other_keys = [k for k in CUSTOM_KEYWORDS.keys() if k != "name"]
    for k in other_keys:
        if k in txt:
            return k, 0.99
    words = re.findall(r"[a-zA-Z']+", txt)
    for w in words:
        m = __import__('difflib').get_close_matches(w, other_keys, n=1, cutoff=0.9)
        if m:
            return m[0], 0.8
    return None, 0.0

def detect_emotion(user_text: str) -> Optional[str]:
    txt = user_text.lower()
    for emo, syns in EMOTION_SYNONYMS.items():
        for s in syns:
            s_lower = s.lower()
            # word boundary check to reduce false positives
            if re.search(rf"\b{re.escape(s_lower)}\b", txt):
                return emo
    return None

# -----------------------
# Session helpers
# -----------------------
def _new_session(title="New Chat"):
    global current_session_id
    sid = uuid.uuid4().hex
    sessions[sid] = {"id": sid, "title": title, "messages": [], "updated": float(time.time())}
    current_session_id = sid
    # ensure user profile exists
    ensure_user_profile(sid)
    return sid

def _current_session():
    global current_session_id
    if current_session_id is None or current_session_id not in sessions:
        sid = uuid.uuid4().hex
        sessions[sid] = {"id": sid, "title": "New Chat", "messages": [], "updated": float(time.time())}
        current_session_id = sid
    return current_session_id



def _update_title_if_needed(sess, user_text):
    if sess["title"] == "New Chat" and user_text.strip():
        clean = re.sub(r"\s+", " ", user_text.strip())
        sess["title"] = (clean[:30] + "…") if len(clean) > 30 else clean

def _trim_session(sess, keep=50):
    if "messages" in sess and len(sess["messages"]) > keep:
        sess["messages"] = sess["messages"][-keep:]
# -----------------------
# Math solver & helpers
# -----------------------
def _safe_number(x):
    try:
        return float(x)
    except Exception:
        return None

def _sympify_arithmetic(expr: str):
    expr = expr.replace('^', '**')
    # treat % as modulo where appropriate
    expr = re.sub(r'(\b[\d\.]+|\b[a-zA-Z]\w*)\s*%\s*(\b[\d\.]+|\b[a-zA-Z]\w*)', r'Mod(\1,\2)', expr)
    allowed_names = {
        "Mod": sp.Mod,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "pi": sp.pi, "E": sp.E,
        "sqrt": sp.sqrt, "ln": sp.log,
        "exp": sp.exp
    }
    try:
        return sp.sympify(expr, locals=allowed_names, convert_xor=True)
    except Exception:
        raise

def _is_safe_token(expr: str):
    return bool(re.fullmatch(r"[0-9A-Za-z\s\+\-\*\/\^\%\(\)\=\.;,\[\]\._]+", expr))

def is_math_expression(text: str) -> bool:
    return bool(re.search(r"[0-9a-zA-Z\+\-\*\/\^\%\(\)=;\., ]", text)) and any(op in text for op in "+-*/^=%()")

def solve_math_expression(expr: str) -> str:
    try:
        # --- Clean unwanted symbols (remove emojis, etc.) ---
        expr = re.sub(r"[^\w\s\[\],().+-/*]", "", expr)
        # --- Preprocess (word → symbol) ---
        replacements = {
            r"\bsquare root of (\w+)\b": r"sqrt(\1)",
            r"\bsquare root of ([0-9]+)\b": r"sqrt(\1)",
            r"\bsquare root of ([a-zA-Z]+)\b": r"sqrt(\1)",
            r"\bsquare root\b": "sqrt",
            r"\bpi\b": "pi",
            r"\be\b": "E",
            r"\^": "**"
        }
        for pat, repl in replacements.items():
            expr = re.sub(pat, repl, expr, flags=re.I)

        # --- System of equations (comma/semicolon separated) ---
        if ";" in expr or "," in expr:
            parts = re.split(r"[;,]", expr)
            equations = []
            symbols = set()
            for part in parts:
                if "=" in part:
                    left, right = part.split("=")
                    eq = sp.Eq(_sympify_arithmetic(left), _sympify_arithmetic(right))
                else:
                    eq = sp.Eq(_sympify_arithmetic(part), 0)
                equations.append(eq)
                symbols.update(eq.free_symbols)
            if symbols:
                sol = sp.solve(equations, list(symbols), dict=True)
                return f"Solution of system: {sol}"


          # --- Matrix detection and operations ---
        if "matrix" in expr.lower():
            try:
                clean_expr = expr.lower().replace("matrix", "sp.Matrix")

                # Evaluate the whole expression safely
                result = eval(clean_expr, {"sp": sp})

                if isinstance(result, sp.Matrix):
                   output = [
                       f"Matrix:\n{result}",
                       f"Determinant: {result.det()}",
                       f"Inverse: {result.inv() if result.det()!=0 else 'Not invertible'}",
                       f"Transpose:\n{result.T}",
                       f"Rank: {result.rank()}",
                       f"Eigenvalues: {result.eigenvals()}",
                     ]
                   return "\n".join(output)
                else:
                   return f"Result:\n{result}"

            except Exception as e:
                return f"Matrix parse error: {e}"

        # --- Plot detection ---
        if expr.lower().startswith("plot"):
           func_exr = expr[4:].strip()
           var = sp.Symbol("x")
           func = sp.sympify(func_expr)
           f_lambdified = sp.lambdify(var, func, "numpy")

           X = np.linspace(-10, 10, 400)
           Y = f_lambdified(X)

           plt.figure()
           plt.plot(X, Y, label=str(func))
           plt.xlabel("x")
           plt.ylabel("f(x)")
           plt.title(f"Graph of {func}")
           plt.legend()
           plt.grid(True)
 
           buf = io.BytesIO()
           plt.savefig(buf, format="png")
           plt.close()
           buf.seek(0)
           return buf 

        # --- Derivative detection ---
        m_der = re.match(r"^\s*(differentiate|derivative)\s+(.+)$", expr, re.I)
        if m_der:
            func_expr = m_der.group(2)
            try:
                # check wrt and order
                wrt_match = re.search(r"w\.r\.t\s+([a-zA-Z])", func_expr)
                order_match = re.search(r"order\s+(\d+)", func_expr)
                var = sp.Symbol(wrt_match.group(1)) if wrt_match else sp.Symbol('x')
                order = int(order_match.group(1)) if order_match else 1
                func_expr_clean = re.sub(r"w\.r\.t\s+[a-zA-Z]", "", func_expr)
                func_expr_clean = re.sub(r"order\s+\d+", "", func_expr_clean)
                func = _sympify_arithmetic(func_expr_clean)
                derivative = sp.diff(func, var, order)
                return f"The {order}-order derivative of {func} w.r.t {var} is: {derivative}"
            except Exception as e:
                return f"Could not differentiate: {e}"

        # --- f(x) = ... style functions ---
        m_func = re.match(r"\s*[a-zA-Z]\w*\(\s*([a-zA-Z]\w*)\s*\)\s*=\s*(.+)", expr)
        if m_func:
            try:
                var_name, func_expr = m_func.groups()
                var = sp.Symbol(var_name)
                func = _sympify_arithmetic(func_expr)
                derivative = sp.diff(func, var)
                return f"The derivative of {func_expr} w.r.t {var_name} is: {derivative}"
            except Exception as e:
                return f"Could not differentiate function: {e}"

        # --- Integral detection ---
        if expr.lower().startswith("integrate"):
            try:
                func_expr = expr[9:].strip()
                wrt_match = re.search(r"w\.r\.t\s+([a-zA-Z])", func_expr)
                var = sp.Symbol(wrt_match.group(1)) if wrt_match else sp.Symbol('x')
                func_expr_clean = re.sub(r"w\.r\.t\s+[a-zA-Z]", "", func_expr)
                func = _sympify_arithmetic(func_expr_clean)
                integral = sp.integrate(func, var)
                return f"Integral of {func} w.r.t {var} is: {integral}"
            except Exception as e:
                return f"Could not integrate: {e}"

               
         # --- Advanced Limit detection ---
        if expr.lower().startswith("limit"):
            try:
                m = re.match(r"limit\s*\(\s*(.+),\s*([a-zA-Z])\s*->\s*([^\s)]+)\s*\)", expr, re.I)
                if not m:
                    m = re.match(r"limit\s+(.+)\s+as\s+([a-zA-Z])\s*->\s*([^\s)]+)", expr, re.I)
                if m:
                    func_expr, var_name, point = m.groups()
                    var = sp.Symbol(var_name)
                    func = sp.sympify(func_expr)
                    dir_flag = "+"
                    if point.endswith("+"): point, dir_flag = point[:-1], "+"
                    if point.endswith("-"): point, dir_flag = point[:-1], "-"
                    point_val = sp.oo if point.lower() in ["oo","inf"] else -sp.oo if point.lower() in ["-oo","-inf"] else sp.sympify(point)
                    return f"Limit of {func} as {var}->{point}{'' if dir_flag=='+' else dir_flag} = {sp.limit(func,var,point_val,dir=dir_flag)}"
            except Exception as e:
                return f"Could not compute limit: {e}"



        # --- Default evaluation ---
        result = _sympify_arithmetic(expr)
        simplified = sp.simplify(result)
        return f"Result: {simplified}"

    except Exception as e:
        return f"🧮 Could not solve equation: {e}"
# -----------------------
# External helpers
# -----------------------
# -------------------------
# GitHub code fetch function
# -------------------------
# Helper to give local answers for basic queries
# --- news fetcher ---
def get_news_search(query=None, max_results=5, lang="en-IN"):
    try:
        if query:
            q = query.replace(" ", "+")
            rss_url = f"https://news.google.com/rss/search?q={q}&hl={lang}&gl=IN&ceid=IN:{lang.split('-')[0]}"
        else:
            rss_url = f"https://news.google.com/rss?hl={lang}&gl=IN&ceid=IN:{lang.split('-')[0]}"

        feed = feedparser.parse(rss_url)
        news_items = []
        for entry in feed.entries[:max_results]:
            title = entry.title
            link = entry.link
            # text-only format
            news_items.append(f"📰 {title}\n👉 {link}")

        return "\n\n".join(news_items) if news_items else "❌ No news found."
    except Exception as e:
        return f"❌ Error fetching news: {e}"

def local_answer(query):
    q = query.lower()
    if "chatgpt" in q:
        return "ChatGPT is an AI language model developed by OpenAI. It can answer questions, generate text and assist in many tasks."
    if "who are you" in q or "what can you do" in q:
        return "I'm your AI assistant. I can answer questions or fetch live results from Google for you."
    return None

# Google Search Helper
def google_search(query, api_key=None, cx_id=None, num_results=5):
   api_key = api_key or GOOGLE_API_KEY
   cx_id = cx_id or GOOGLE_CX
   if not api_key or not cx_id:
      return []
   try: 
      url = "https://www.googleapis.com/customsearch/v1"
      params = {"key": api_key, "cx": cx_id, "q": query, "num": num_results}
      r = requests.get(url, params=params, timeout=10)
      data = r.json()
      return [{"title": i.get("title"), "link": i.get("link"), "snippet": i.get("snippet")} for i in data.get("items", [])]
   except Exception:
      return []


    # Google Image Search Helper
def google_image_search(query, api_key=None, cx_id=None, num_results=3):
     api_key = api_key or GOOGLE_API_KEY
     cx_id = cx_id or GOOGLE_CX
     try:
          url = "https://www.googleapis.com/customsearch/v1"
          params = {"key": api_key, "cx": cx_id, "q": query, "searchType": "image", "num": num_results}
          r = requests.get(url, params=params, timeout=10)
          return [i.get("link") for i in r.json().get("items", [])]
     except Exception:
          return []
def fetch_code_from_github(query, language="python", max_results=1):
    """
    Fetch code from GitHub by keyword query.

    query: search keyword (e.g., "add two numbers")
    language: programming language (python, java, cpp, etc.)
    max_results: number of results to fetch
    """
    url = f"https://api.github.com/search/code?q={query}+language:{language}&per_page={max_results}"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 401:
            return "API error: 401 Unauthorized. Check your token."
        elif response.status_code != 200:
            return f" API error: {response.status_code}"

        data = response.json()
        if "items" not in data or len(data["items"]) == 0:
            return "No code found for your query."

        item = data["items"][0]
        file_url = item.get("html_url", "")
        repo_info = item.get("repository", {})
        raw_url_base = repo_info.get("html_url", "").replace("github.com", "raw.githubusercontent.com")
        file_path = item.get("path", "")
        branch = repo_info.get("default_branch", "main")  # fallback to 'main'

        if not raw_url_base or not file_path:
            return f"Code found here: {file_url}"

        raw_file_url = f"{raw_url_base}/{branch}/{file_path}"
        raw_resp = requests.get(raw_file_url, headers=headers, timeout=10)

        if raw_resp.status_code == 200:
            return raw_resp.text
        else:
            return f"Code found here: {file_url} (could not fetch raw file)"

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
    
def handle_code_request(user_message):
    # Language detect
    language = "python"
    msg = user_message.lower() 
    if "java" in msg:
      language = "java"
    elif "c++" in msg or "cpp" in msg:
      language = "cpp"
    elif "javascript" in msg or "js" in msg:
      language = "javascript"
    elif "python" in msg:
      language = "python"
    elif "c" in msg:
      language = "c"
    # Program keyword detect
    if "program" in msg or "code" in msg or "example" in msg:
      query = user_message.replace("send me", "").replace("program", "").strip()
      return fetch_code_from_github(query, language=language, max_results=1)
    return None

def highlight_keywords(answer: str, question: str) -> str:
    """
    Highlights words from the user's question in the chatbot's answer.
    """
    keywords = [word for word in re.findall(r"\w+", question) if len(word) > 2]
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        answer = pattern.sub(
            lambda m: f"<mark style='background-color:yellow; font-weight:bold'>{m.group(0)}</mark>",
            answer,
        )
    return answer
def search_youtube_link(query: str):
    if not YOUTUBE_API_KEY:
        return None, "YouTube API key missing"

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        req = youtube.search().list(
            q=query,
            part="id,snippet",
            type="video",
            maxResults=3   # 👈 3 videos instead of 1
        )
        res = req.execute()
        items = res.get("items", [])
        if not items:
            return None, "No results"

        # Collect up to 3 video URLs
        video_urls = [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in items]
        return video_urls, None

    except HttpError as e:
        if e.resp.status == 403:
            return None, "YouTube quota exceeded, try again tomorrow"
        return None, f"YouTube API error: {e}"
    except Exception as e:
        return None, f"YouTube search failed: {e}"

def get_wikipedia_answer(query, lang="en"):
    try:
        wikipedia.set_lang(lang)
        mapped_query = preprocess_for_wikipedia(query)
        page = wikipedia.page(mapped_query)
        content = page.summary
        sentences = content.split('. ')
        bullet_points = [f"•  {s.strip('.')}" for s in sentences if s.strip()]
        return "\n".join(bullet_points)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Try being more specific. Suggestions: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Sorry! Couldn't find anything."
    except Exception as e:
        app.logger.exception("Wikipedia error: %s", e)
        return f"Error! {e}"

def get_duckduckgo_summary(query):
    try:
        url = f"https://api.duckduckgo.com/?q={requests.utils.requote_uri(query)}&format=json"
        res = requests.get(url, timeout=10)
        data = res.json()
        abstract = data.get("Abstract")
        return f"📚 DuckDuckGo: {abstract}" if abstract else "❌ Couldn't find information"
    except Exception as e:
        app.logger.exception("DuckDuckGo error: %s", e)
        return f"❌ Error! {e}"

def get_geeksforgeeks_answer(query):
    try:
        search_url = f"https://www.geeksforgeeks.org/?s={requests.utils.quote(query)}"
        search_response = requests.get(search_url, timeout=10)
        search_soup = BeautifulSoup(search_response.text, 'html.parser')
        first_link = None
        card = search_soup.select_one('a.cc-card, a.CardsList__link, a[data-gfg-action="click"]')
        if card and card.get("href"):
            first_link = card["href"]
        if not first_link:
            container = search_soup.find('div', class_='articles-list')
            if container:
                a = container.find('a', href=True)
                if a:
                    first_link = a['href']
        if not first_link:
            return "❌ Couldn't find relevant Answer"
        article_url = first_link
        article_response = requests.get(article_url, timeout=10)
        article_soup = BeautifulSoup(article_response.text, 'html.parser')
        paragraphs = article_soup.find_all('p')
        content = ' '.join(p.get_text(strip=True) for p in paragraphs[:3]).strip()
        return f"📘 GeeksforGeeks:\n{content}\n\n🔗 Full program: {article_url}"
    except Exception as e:
        app.logger.exception("GfG fetch error: %s", e)
        return f"❌ Error! {e}"

def split_text_and_links(answer: str):
    """Split first paragraph and all links"""
    parts = answer.split("http")
    if len(parts) == 1:
        return answer.strip(), []
    text = parts[0].strip()
    links = ["http" + p.strip() for p in parts[1:]]
    return text, links

def format_response(text: str, links: list):
    """Format response so that paragraph is first, then links separately"""
    formatted = text
    if links:
        formatted += "<br><br><b>Sources:</b><br>"
        for link in links:
            formatted += f"- <a href='{link}' target='_blank'>{link}</a><br>"
    return formatted


# -----------------------
# Cartoonify
# -----------------------
def _safe_image_filename(name: str) -> str:
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    return safe_name
# -----------------------
# Routes
# -----------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True})

@app.route("/", methods=['GET', 'POST'])
@login_required
def index():
    return render_template("index.html")
# -----------------------
# New Chat (DB session create)
# -----------------------
@app.route("/new_chat", methods=["POST"])
@login_required
def new_chat():
    # DB create new ChatSession
    new_session = ChatSession(user_id=current_user.id, title="New Chat")
    db.session.add(new_session)
    db.session.commit()

    # Update current session id
    globals()["current_session_id"] = new_session.id

    return jsonify({"status": "new chat started", "session_id": new_session.id})
#--------------------------------------
@app.route("/code", methods=["POST"])
def get_code():
    user_message = request.json.get("message", "")
    language = "python"  # default
    msg = user_message.lower()
    if "java" in msg: language = "java"
    elif "c++" in msg: language = "cpp"
    elif "javascript" in msg or "js" in msg: language = "javascript"
    
    query = user_message.replace("send me", "").replace("program", "").strip()
    code_result = fetch_code_from_github(query, language=language)
    return jsonify({"response": code_result})

@app.route("/math", methods=["POST"])
@login_required
def math():  # def chat -> def math
    # --- accept JSON or form ---
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or request.form.get("message") or "").strip()
    provided_session_id = (data.get("session_id") or request.form.get("session_id") or "").strip()

    if not user_message:
        return jsonify({"response": "Empty message received.", "source": "system"}), 400

    curr_sid = _current_session()
    if provided_session_id:
        sid = provided_session_id
        if sid not in sessions:
            sessions[sid] = {"id": sid, "title": "New Chat", "messages": [], "updated": float(time.time())}
        globals()['current_session_id'] = sid
    else:
        sid = curr_sid
        if sid not in sessions:
            sessions[sid] = {"id": sid, "title": "New Chat", "messages": [], "updated": float(time.time())}

    sess = sessions[sid]
    user_id_for_profile = sid  

    base_response = {
        "response": None,
        "source": "system",
        "image_url": None,
        "session_id": sid
    }

    # Title update & preview
    bot_preview = user_message
    if chat_histories and not chat_histories[-1].get("saved", False):
        chat_histories[-1]["messages"].append({"role": "user", "content": user_message})
        chat_histories[-1]["messages"].append({"role": "bot", "content": bot_preview})
    else:
        chat_histories.append({
            "title": user_message,
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "bot", "content": bot_preview}
            ],
            "saved": False,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    _update_title_if_needed(sess, user_message)
    um_lower = user_message.lower()

    ensure_user_profile(user_id_for_profile)

    # **Extra: factorial & permutations detection**
    if re.match(r"^factorial\s+\d+$", um_lower):
        try:
            n = int(user_message.split()[-1])
            return jsonify({"response": f"{n}! = {sp.factorial(n)}", "session_id": sid})
        except Exception as e:
            return jsonify({"response": f"Could not compute factorial: {e}", "session_id": sid})

    if re.match(r"^ncr\s+\d+\s+\d+$", um_lower):
        try:
            _, n, r = user_message.split()
            n, r = int(n), int(r)
            return jsonify({"response": f"C({n},{r}) = {sp.binomial(n, r)}", "session_id": sid})
        except Exception as e:
            return jsonify({"response": f"Could not compute nCr: {e}", "session_id": sid})

    if re.match(r"^npr\s+\d+\s+\d+$", um_lower):
        try:
            _, n, r = user_message.split()
            n, r = int(n), int(r)
            return jsonify({"response": f"P({n},{r}) = {sp.factorial(n)/sp.factorial(n-r)}", "session_id": sid})
        except Exception as e:
            return jsonify({"response": f"Could not compute nPr: {e}", "session_id": sid})
        
    # --- normalize math input ---
    replacements = {
        r"\bsinx\b": "sin(x)",
        r"\bcosx\b": "cos(x)",
        r"\btanx\b": "tan(x)",
        r"\bcotx\b": "cot(x)",
        r"\bsecx\b": "sec(x)",
        r"\bcscx\b": "csc(x)",
        r"\bsqrtx\b": "sqrt(x)",
    }
    for pattern, repl in replacements.items():
        user_message = re.sub(pattern, repl, user_message, flags=re.IGNORECASE)
    um_lower = user_message.lower()

    # --- Equation solving (prefix: solve ...) ---
    if um_lower.startswith("solve"):
        try:
            expr_str = user_message[5:].strip()
            x = sp.Symbol('x')
            if "=" in expr_str:
                left, right = expr_str.split("=")
                expr = sp.sympify(left) - sp.sympify(right)
            else:
                expr = sp.sympify(expr_str)
            sol = sp.solve(expr, x)
            return jsonify({"response": f"Solutions: {sol}", "session_id": sid})
        except Exception as e:
            return jsonify({"response": f"Could not solve equation: {e}", "session_id": sid})

    # --- Plotting (prefix: plot ...) ---
    if um_lower.startswith("plot"):
        try:
            raw = user_message[4:].strip()
            parts = re.split(r'\s*[;,]\s*', raw)
            x = sp.Symbol('x')
            curves = []
            for p in parts:
                if not p:
                    continue
                s = re.sub(r'(?i)f\s*\(\s*x\s*\)\s*=', '', p).replace('^', '**')
                s = re.sub(r'(?<=\d)\s*(?=x\b)', '*', s)
                expr = sp.sympify(s)
                curves.append((s, expr))
            if not curves:
                return jsonify({"response": "Plot: no expressions found.", "session_id": sid})

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np, uuid, os

            xs = np.linspace(-10, 10, 400)
            plt.figure()
            for label, expr in curves:
                f = sp.lambdify(x, expr, "numpy")
                ys = f(xs)
                plt.plot(xs, ys, label=label)
            plt.xlabel("x")
            plt.ylabel("f(x)")
            if len(curves) > 1:
                plt.legend()
            plt.grid(True)

            img_id = uuid.uuid4().hex
            filename = f"plot_{img_id}.png"
            plot_dir = os.path.join("static", "plots")
            os.makedirs(plot_dir, exist_ok=True)   # auto create folder if not exists
            img_path = os.path.join(plot_dir, filename)
            plt.savefig(img_path) 
            plt.close()

            return jsonify({
                "response": f"Here is the plot of {', '.join(lbl for lbl, _ in curves)}",
                "result": None,
                "image_url": url_for("static", filename=f"plots/{filename}"),
               "session_id": sid
            })
        except Exception as e:
            return jsonify({"response": f"Could not plot: {e}", "session_id": sid})

    # --- Simplify (prefix: simplify ...) ---
    if um_lower.startswith("simplify"):
        try:
            expr_str = user_message[8:].strip()
            expr = sp.sympify(expr_str)
            simplified = sp.simplify(expr)
            return jsonify({"response": f"Result: {simplified}", "session_id": sid})
        except Exception as e:
            return jsonify({"response": f"Could not simplify: {e}", "session_id": sid})

    # --- Matrix detection & operations ---
    if "matrix" in um_lower:
        try:
            clean_expr = user_message.lower().replace("matrix", "sp.Matrix")
            result = eval(clean_expr, {"sp": sp})

            if isinstance(result, sp.Matrix):
                output = [
                    f"Matrix:\n{result}",
                    f"Determinant: {result.det()}",
                    f"Inverse: {result.inv() if result.det()!=0 else 'Not invertible'}",
                    f"Transpose:\n{result.T}",
                    f"Rank: {result.rank()}",
                    f"Eigenvalues: {result.eigenvals()}",
                ]
                reply = "\n".join(output)
            else:
                reply = f"Result:\n{result}"

            sess["messages"] += [{"role": "user", "content": user_message}, {"role": "bot", "content": reply}]
            _trim_session(sess)
            sess["updated"] = float(time.time())
            return jsonify({"response": reply, "session_id": sid})

        except Exception as e:
            return jsonify({"response": f"Matrix parse error: {e}", "session_id": sid})

    if is_math_expression(user_message) and _is_safe_token(user_message): # --- Math detection ---
     try:
        math_keywords = ["+", "-", "*", "/", "square root", "√", "sqrt", "^", "log", "sin", "cos", "tan"]
        if any(k in um_lower for k in math_keywords) or is_math_expression(user_message):
            math_answer = solve_math_expression(user_message)  
            if math_answer:
                sess["messages"] += [
                    {"role": "user", "content": user_message},
                    {"role": "bot", "content": f"🧮 {math_answer}"}
                ]
                _trim_session(sess)
                sess["updated"] = float(time.time())
                return jsonify({"response": f"🧮 {math_answer}", "session_id": sid})
     except Exception:
        app.logger.exception("Error during math detection/solve.")

@app.route("/news", methods=["POST"])
def get_news():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    provided_session_id = (data.get("session_id") or "").strip()

    if not user_message:
        return jsonify({"response": "Empty message received.", "source": "system"}), 400

    # --- session handling ---
    sid = provided_session_id or _current_session()
    if sid not in sessions:
        sessions[sid] = {"id": sid, "title": "", "messages": [], "updated": float(time.time())}

    sess = sessions[sid]

    # --- add user message ---
    sess["messages"].append({"role": "user", "content": user_message})
    _update_title_if_needed(sess, user_message)

    # --- detect news keywords ---
    um_lower = user_message.lower()
    news_keywords = ["news", "செய்தி", "seithi", "seithigal", "vartaigal"]

    if any(word in um_lower for word in news_keywords):
        news_response = get_news_search(query=user_message, lang="en-IN")
        sess["messages"].append({"role": "bot", "content": news_response})
        _trim_session(sess)
        sess["updated"] = float(time.time())
        return jsonify({"response": news_response, "session_id": sid})

    # fallback
    return jsonify({"response": "No news detected in your message.", "session_id": sid})


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    # --- accept JSON or form ---
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or request.form.get("message") or "").strip()

    if not user_message:
        return jsonify({"response": "Empty message received.", "source": "system"}), 400

    # --- resolve session_id properly ---
    provided_session_id = data.get("session_id") or request.form.get("session_id")

    if provided_session_id:
      db_session = ChatSession.query.filter_by(id=provided_session_id, user_id=current_user.id).first()
      if not db_session:
        db_session = ChatSession(user_id=current_user.id, title="New Chat")
        db.session.add(db_session)
        db.session.commit()
      sid = db_session.id
    else:
      sid = globals().get("current_session_id")
      if not sid:
        db_session = ChatSession(user_id=current_user.id, title="New Chat")
        db.session.add(db_session)
        db.session.commit()
        sid = db_session.id
        globals()["current_session_id"] = sid


    sess = sessions[sid]
    user_id_for_profile = sid  # keep your existing user-profile-by-session behavior

    base_response = {
        "response": None,
        "source": "system",
        "image_url": None,
        "session_id": sid
    }

    # --- chat title / preview history (your existing UI preview logic) ---
    bot_preview = user_message
    if chat_histories and not chat_histories[-1].get("saved", False):
        chat_histories[-1]["messages"].append({"role": "user", "content": user_message})
        chat_histories[-1]["messages"].append({"role": "bot", "content": bot_preview})
    else:
        chat_histories.append({
            "title": user_message,
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "bot", "content": bot_preview}
            ],
            "saved": False,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    _update_title_if_needed(sess, user_message)
    um_lower = user_message.lower()

    # --- ensure "learning agent" profile exists ---
    ensure_user_profile(user_id_for_profile)

    # --- Language detection (ta/en) ---
    try:
        lang = detect(user_message)
        if re.search(r'[\u0B80-\u0BFF]', user_message):
            lang = 'ta'
    except Exception:
        lang = "unknown"
    if lang.startswith("ta"):
        set_user_language(user_id_for_profile, "ta")
    elif lang.startswith("en"):
        set_user_language(user_id_for_profile, "en")

    # --- Greetings ---
    greetings_map = {
        ('hi', 'hii', 'hello', 'excuse me', 'hlo'): "Hello! 😊 How can I help you today?",
        ('வணக்கம்', 'vanakkam', 'vanakam'): "வணக்கம்! நான் உங்களுக்கு உதவ வேண்டுமா?",
        ("how are you?", "how are you", "how are u"): "I am fine! 😊 What about you?",
        ('fine', 'ok', 'thanks', 'thank you', 'thank u'): "😊",
        ("You are so good","you are very nice","you are great"):"🥰Aww! It's My pleasure"
    }
    for keys, g_reply in greetings_map.items():
        if um_lower in keys:
            sess["messages"] += [{"role": "user", "content": user_message}, {"role": "bot", "content": g_reply}]
            _trim_session(sess)
            sess["updated"] = float(time.time())
            return jsonify({"response": g_reply, "session_id": sid})

    # --- Intent / Emotion ---
    intent, intent_conf = detect_intent(user_message)
    if intent and intent_conf >= 0.75:
        i_reply = random.choice(CUSTOM_KEYWORDS.get(intent, [f"I am {BOT_NAME}."]))
        sess["messages"] += [{"role": "user", "content": user_message}, {"role": "bot", "content": i_reply}]
        _trim_session(sess)
        sess["updated"] = float(time.time())
        return jsonify({"response": i_reply, "session_id": sid})

    emo = detect_emotion(user_message)
    if emo:
        try:
            update_user_emotion(user_id_for_profile, emo)
        except Exception:
            pass
        emo_reply = random.choice(EMOTION_REPLIES.get(emo, EMOTION_REPLIES.get("neutral")))
        sess["messages"] += [{"role": "user", "content": user_message}, {"role": "bot", "content": emo_reply}]
        _trim_session(sess)
        sess["updated"] = float(time.time())
        return jsonify({"response": emo_reply, "session_id": sid})

    # --- Quick knowledge / math fallback before LLM ---
    if "who" in um_lower or "what" in um_lower:
        bot_response = get_wikipedia_answer(user_message)
    elif any(op in um_lower for op in ["+", "-", "*", "/", "sin", "cos", "tan"]):
        bot_response = solve_math_expression(user_message)
    else:
        bot_response = "I'm still learning. Can you rephrase?"

    bot_response = highlight_keywords(bot_response, user_message)

    # --- code request (GitHub) ---
    code_result = handle_code_request(user_message)
    if code_result:
        sess["messages"].append({"role": "user", "content": user_message})
        sess["messages"].append({"role": "bot", "content": code_result})
        _trim_session(sess)
        sess["updated"] = float(time.time())
        return jsonify({"response": code_result, "session_id": sid})
    
    # --- Web answers (Wiki/DDG/GfG) ---
    handled = False
    response_text = None

    local_resp = local_answer(user_message)
    if local_resp:
       sess["messages"].append({"role": "assistant", "content": local_resp})
       return jsonify({"response": local_resp})
    if "image" in user_message.lower() or "picture" in user_message.lower():
       image_links = google_image_search(user_message)
       if image_links:
         # Just send **one image url** (or many in list)
         return jsonify({
            "image_url": image_links[0],   # first image
            "response": None               # no text response
        })
       else:
         return jsonify({"response": "Sorry, no image found."})


    google_results = google_search(user_message)
    if google_results:
      # only snippet highlight
      paragraph = " ".join([
        highlight_keywords(i['snippet'], user_message)
        for i in google_results if i.get("snippet")
    ])

      links = "\n".join([
        f"• {highlight_keywords(i['title'], user_message)}: {i['link']}"
        for i in google_results if i.get("title") and i.get("link")
    ])

      formatted = (
        f"You asked: {user_message}\n\n"
        f"{paragraph}\n\n"
        f"Sources:\n{links}"
    )

      sess["messages"].append({"role": "assistant", "content": formatted})
      return jsonify({"response": formatted})

        # 5) Google Image Search
    if "image" in user_message.lower() or "picture" in user_message.lower():
        image_links = google_image_search(user_message)
        if image_links:
            formatted = "\n".join(image_links)
            sess["messages"].append({"role": "assistant", "content": formatted})
            return jsonify({"response": formatted})

    wiki_answer = get_wikipedia_answer(user_message, lang='en')
    if wiki_answer and not wiki_answer.startswith(("Error", "Sorry", "Ambiguous")):
       # --- assume wiki_answer return panna text la links irukum ---
       main_text, links = split_text_and_links(wiki_answer)
       response_text = format_response(main_text, links)
       handled = True
    else:
       ddg = get_duckduckgo_summary(user_message)
       if ddg and not ddg.startswith("❌"):
         main_text, links = split_text_and_links(ddg)
         response_text = format_response(main_text, links)
         handled = True
       else:
         gfg = get_geeksforgeeks_answer(user_message)
         if gfg:
            main_text, links = split_text_and_links(gfg)
            response_text = format_response(main_text, links)
            handled = True

    if handled:
        sess["messages"] += [{"role": "user", "content": user_message}, {"role": "bot", "content": response_text}]
        _trim_session(sess)
        sess["updated"] = float(time.time())
        update_user_topic(user_id_for_profile, "knowledge")
        return jsonify({"response": response_text, "session_id": sid})

    # --- GPT4All Tanglish + English fallback ---
    try:
        # give full context to model
        reply = tanglish_reply(sess, user_message, max_tokens=512)
    except Exception as e:
        app.logger.exception("Tanglish+English GPT fallback failed: %s", e)
        reply = "Sorry, something went wrong during response generation."

    # --- Save to DB (both user and bot) ---
    try:
      user_chat = ChatHistory(user_id=current_user.id, session_id=sid, role="user", content=user_message)
      bot_chat  = ChatHistory(user_id=current_user.id, session_id=sid, role="bot",  content=reply)
      db.session.add(user_chat)
      db.session.add(bot_chat)

    # update title if it's the first msg
      session_obj = ChatSession.query.get(sid)
      if session_obj and session_obj.title == "New Chat":
        session_obj.title = user_message[:30]

      db.session.commit()
    except Exception as e:
      db.session.rollback()
      app.logger.exception("DB save failed: %s", e)


    # --- update in-memory session + return ---
    sess["messages"].append({"role": "bot", "content": reply})
    _trim_session(sess)
    sess["updated"] = float(time.time())

    base_response["response"] = reply
    base_response["source"] = "chat"
    return jsonify(base_response)


@app.route('/search')
def search():
    query = request.args.get('q', '')
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={query}"
    r = requests.get(url)
    data = r.json()

    if 'items' not in data:
        return jsonify({"summary": "", "links": []})

    snippets = " ".join(item['snippet'] for item in data['items'])
    summary = ". ".join(snippets.split(". ")[:10]) + "."
    links = [item['link'] for item in data['items']]

    return jsonify({"summary": summary, "links": links})

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            return "Username already exists!"
        if User.query.filter_by(email=email).first():
            return "Email already exists!"
        
        # Hash the password correctly
        hashed_pw = generate_password_hash(password, method="pbkdf2:sha256")

        # Create new user with email
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")  # Use email instead of username
        password = request.form.get("password")

        if not email or not password:
            flash("Please enter both email and password!", "error")
            return render_template("login.html")

        user = User.query.filter_by(email=email).first()  # Query by email (unique)
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("index"))

        flash("Invalid email or password!", "error")
        return render_template("login.html")

    return render_template("login.html")



@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---- Session APIs (sidebar) ----
# -----------------------
# List all Sessions for Sidebar
# -----------------------
@app.route("/sessions", methods=["GET"])
@login_required
def list_sessions():
    chats = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.updated.desc()).all()
    return jsonify([
        {"id": c.id, "title": c.title, "updated": c.updated.isoformat() if c.updated else None}
        for c in chats
    ])

@app.route("/session/<sid>", methods=["GET"])
def get_session_messages_page(sid):
    if sid not in sessions:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"id": sid, "title": sessions[sid]["title"], "messages": sessions[sid]["messages"]})

@app.route("/session/<sid>/activate", methods=["POST"])
def activate_session(sid):
    global current_session_id
    if sid not in sessions:
        return jsonify({"error": "session not found"}), 404
    current_session_id = sid
    return jsonify({"ok": True, "current": current_session_id})

@app.route("/api/new_session", methods=["POST"])
@login_required
def new_session():
    new_sess = ChatSession(user_id=current_user.id, title="New Chat")
    db.session.add(new_sess)
    db.session.commit()
    return jsonify({"session_id": new_sess.id})

  
# ---- Chat APIs ----

@app.route("/api/save", methods=["POST"])
@login_required
def save_message():
    data = request.json
    session_id = data.get("session_id")
    role = data.get("role")
    content = data.get("content")

    # Message save
    msg = ChatHistory(user_id=current_user.id, session_id=session_id, role=role, content=content)
    db.session.add(msg)

    if not session_id:
        new_session = ChatSession(user_id=current_user.id, title="Auto-created")
        db.session.add(new_session)
        db.session.commit()
        session_id = new_session.id

    # If first user message, update session title
    session = ChatSession.query.get(session_id)
    if role == "user" and session.title == "New Chat":
        session.title = content[:50]  # first 50 chars
        db.session.add(session)

    db.session.commit()
    return jsonify({"status": "ok"})



@app.route("/history", defaults={"sid": None}, methods=["GET"])
@app.route("/history/<int:sid>", methods=["GET"])
@login_required
def get_history(sid):
    if sid:
        # ஒரே session
        chats = ChatHistory.query.filter_by(
            session_id=sid, user_id=current_user.id
        ).order_by(ChatHistory.timestamp).all()
        return jsonify([
            {"role": c.role, "content": c.content, "time": c.timestamp.isoformat()}
            for c in chats
        ])
    else:
        # எல்லா sessions
        sessions = ChatSession.query.filter_by(
            user_id=current_user.id
        ).order_by(ChatSession.created_at.desc()).all()

        session_list = []
        for s in sessions:
            msgs = ChatHistory.query.filter_by(session_id=s.id).order_by(ChatHistory.timestamp).all()
            preview = msgs[0].content if msgs else ""
            session_list.append({
                "id": s.id,
                "title": s.title,
                "preview": preview[:50] + ("..." if len(preview) > 50 else ""),
                "messages": [{"role": m.role, "content": m.content} for m in msgs]
            })

        return jsonify({
            "user": {"name": current_user.username},
            "sessions": session_list
        })

@app.route("/api/history", methods=["GET"])
@login_required
def history():
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).all()
    result = []
    for s in sessions:
        # get only the last USER message
        last_user_msg = ChatHistory.query.filter_by(session_id=s.id, role="user")\
                                         .order_by(ChatHistory.timestamp.desc()).first()
        result.append({
            "id": s.id,
            "title": s.title,
            "last_message": f"You asked: {last_user_msg.content}" if last_user_msg else ""
        })
    return jsonify({"sessions": result})


@app.route("/api/messages/<int:session_id>", endpoint="get_session_messages_api")
@login_required
def get_session_messages_api(session_id):
    messages = ChatHistory.query.filter_by(session_id=session_id).order_by(ChatHistory.timestamp).all()
    return jsonify({"messages": [{"content": m.content, "role": m.role} for m in messages]})

@app.route("/api/session/<int:sid>", methods=["DELETE"])
@login_required
def delete_session(sid):
    session_obj = ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()
    if not session_obj:
        return jsonify({"error": "Session not found"}), 404

    # delete messages first
    ChatHistory.query.filter_by(session_id=sid).delete()
    db.session.delete(session_obj)
    db.session.commit()
    return jsonify({"status": "ok"})
@app.route("/api/history/delete_all", methods=["DELETE"])
@login_required
def delete_all_history():
    try:
        # Delete all messages for this user
        ChatHistory.query.filter_by(user_id=current_user.id).delete()
        # Delete all sessions
        ChatSession.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        db.session.rollback()
        app.logger.exception("Failed to delete all history: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500

# Health endpoint for user profile (learning agent) quick check
@app.route('/user/<uid>/profile', methods=['GET'])
def get_user_profile(uid):
    users = load_users()
    if uid not in users:
        return jsonify({"error": "user not found"}), 404
    return jsonify(users[uid])

# -----------------------
# Boot
# -----------------------
if __name__ == "__main__":
    _new_session()
    with app.app_context():
        db.create_all()   
    # In production, run with gunicorn/uvicorn and set debug=False
    app.run(debug=True, host="0.0.0.0", port=5000)