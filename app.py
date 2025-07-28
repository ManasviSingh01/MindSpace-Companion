"""
MindSpace Companion – full, fixed, and feature-complete
- Improved emotion detection (text + voice)
- Working analytics tab
- PDF upload & auto mood detection
- AI-ish weekly summaries
- Calendar (.ics) export for entries + daily reminders

Run:  streamlit run app.py
Reqs: pip install -U streamlit vaderSentiment PyPDF2 scikit-learn wordcloud altair matplotlib numpy pandas librosa sounddevice soundfile
"""

import os
import io
import json
import random
import calendar
import datetime as dt
from datetime import timedelta
import re
import unicodedata

import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components

import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ========= OPTIONAL / BEST-EFFORT IMPORTS =========
AUDIO_AVAILABLE, LIBROSA_AVAILABLE, PDF_AVAILABLE = False, False, False
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

try:
    import librosa
    from librosa.feature import rms as librosa_rms
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# =================== CONFIG & CONSTANTS ===================

EMOJI = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "anxious": "😰", "neutral": "😐", "gratitude": "🙏"
}

MOTIVATIONAL_QUOTES = {
    "happy": ["Shine bright!", "Your joy is radiant!", "Stay awesome!"],
    "sad": ["This too shall pass.", "You're not alone.", "Feel it, release it."],
    "angry": ["Breathe. You’re in control.", "It’s okay to pause.", "Stay grounded."],
    "anxious": ["You are safe.", "Breathe in peace.", "Let go of the worry."],
    "neutral": ["Balance is strength.", "A quiet day is progress too.", "Stay steady."],
    "gratitude": ["Gratitude grows joy.", "Cherish small moments.", "You're blessed."]
}

SPOTIFY_PLAYLISTS = {
    "happy": "37i9dQZF1DXdPec7aLTmlC",
    "sad": "37i9dQZF1DX7qK8ma5wgG1",
    "angry": "37i9dQZF1DX3YSRoSdA634",
    "anxious": "37i9dQZF1DX9uKNf5jGX6m",
    "neutral": "37i9dQZF1DX4WYpdgoIcn6",
    "gratitude": "37i9dQZF1DX7K31D69s4M1"
}

# Free static GIFs (no API keys)
MOOD_GIF_LINKS = {
    "happy": "https://media.giphy.com/media/111ebonMs90YLu/giphy.gif",
    "sad": "https://media.giphy.com/media/ROF8OQvDmxytW/giphy.gif",
    "angry": "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif",
    "anxious": "https://media.giphy.com/media/l0HlQ7LRalG0vFfO0/giphy.gif",
    "neutral": "https://media.giphy.com/media/xT9Igl6oHnUZfgN4d6/giphy.gif",
    "gratitude": "https://media.giphy.com/media/l41Yg5zK5x7Di3nza/giphy.gif"
}

LOG_FILE = "mood_log.json"
VOICE_DEBUG = False  # toggle in UI

st.set_page_config(page_title="MindSpace Companion", layout="wide")
st.markdown(
    """
<style>
.stApp { background-color: #e0f7fa; }
section[data-testid="stSidebar"] { background-color: #ffe6f0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =================== STATE ===================

if "logs" not in st.session_state:
    st.session_state.logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                st.session_state.logs = json.load(f)
        except Exception:
            st.session_state.logs = []

if "analyzer" not in st.session_state:
    st.session_state.analyzer = SentimentIntensityAnalyzer()

# =================== EMOTION DETECTION (TEXT) ===================

ANGRY_WORDS = {"angry", "furious", "rage", "irritated", "annoyed", "mad"}
ANXIOUS_WORDS = {"anxious", "anxiety", "worried", "nervous", "panic", "overwhelmed"}
GRATITUDE_WORDS = {"grateful", "thankful", "gratitude", "blessed", "appreciate"}
HAPPY_WORDS = {"happy", "joy", "excited", "glad", "cheerful"}
SAD_WORDS = {"sad", "depressed", "unhappy", "down", "miserable"}

# Pre-compiled regexes
RE_ANGRY = re.compile(r"\b(angry|furious|rage|irritated|annoyed|mad)\b", re.I)
RE_ANXIOUS = re.compile(r"\b(anxious|anxiety|worried|nervous|panic|overwhelmed)\b", re.I)
RE_GRATITUDE = re.compile(r"\b(grateful|thankful|gratitude|blessed|appreciate)\b", re.I)
RE_HAPPY = re.compile(r"\b(happy|joy|excited|glad|cheerful)\b", re.I)
RE_SAD = re.compile(r"\b(sad|depressed|unhappy|down|miserable)\b", re.I)

def _intensity_heuristics(text: str) -> float:
    """Crude 'emotional intensity' score based on punctuation, ALLCAPS, elongations."""
    if not text:
        return 0.0
    excls = text.count("!")
    qmarks = text.count("?")
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
    elong = len(re.findall(r"(.)\1{2,}", text))  # sooooo
    return 0.4 * excls + 0.3 * qmarks + 2.0 * caps_ratio + 0.2 * elong

def detect_text_emotion(text: str) -> str:
    """Hybrid rule-based + VADER sentiment detection with intensity heuristics."""
    if not text or not text.strip():
        return "neutral"

    text_lower = text.lower()

    # 1) Explicit emotion keywords first
    if RE_ANGRY.search(text_lower):
        return "angry"
    if RE_ANXIOUS.search(text_lower):
        return "anxious"
    if RE_GRATITUDE.search(text_lower):
        return "gratitude"
    if RE_HAPPY.search(text_lower):
        return "happy"
    if RE_SAD.search(text_lower):
        return "sad"

    # 2) VADER + intensity tweak
    scores = st.session_state.analyzer.polarity_scores(text)
    comp = scores["compound"]
    intensity = _intensity_heuristics(text)

    if comp >= 0.55:
        return "happy"
    elif comp <= -0.45:
        if intensity > 1.0 or any(w in text_lower for w in ANGRY_WORDS):
            return "angry"
        return "sad"
    else:
        if RE_ANXIOUS.search(text_lower):
            return "anxious"
        return "neutral"

# =================== EMOTION DETECTION (VOICE) ===================

def safe_mean(x):
    try:
        return float(np.mean(x)) if x is not None and len(np.atleast_1d(x)) else 0.0
    except Exception:
        return 0.0

def analyze_voice(audio_bytes: bytes) -> str:
    """Lightweight heuristic voice emotion detection using librosa features."""
    if not (AUDIO_AVAILABLE and LIBROSA_AVAILABLE):
        return "neutral"
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        if y.size == 0:
            return "neutral"

        y = librosa.util.normalize(y)

        # Features with safe fallbacks
        try:
            rms = safe_mean(librosa_rms(y=y))
        except Exception:
            rms = safe_mean(np.abs(y))
        try:
            zcr = safe_mean(librosa.feature.zero_crossing_rate(y))
        except Exception:
            zcr = 0.0
        try:
            sc = safe_mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        except Exception:
            sc = 0.0
        try:
            tempo_arr = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
            tempo = float(np.mean(tempo_arr)) if tempo_arr is not None and tempo_arr.size else 0.0
        except Exception:
            tempo = 0.0
        try:
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            f0 = f0[np.isfinite(f0)]
            pitch_range = (np.percentile(f0, 95) - np.percentile(f0, 5)) if f0.size else 0.0
        except Exception:
            pitch_range = 0.0

        if VOICE_DEBUG:
            st.write({"rms": rms, "tempo": tempo, "zcr": zcr, "centroid": sc, "pitch_range": pitch_range})

        # Heuristic thresholds (tweak to your dataset)
        loud = rms > 0.06
        fast = tempo > 110
        harsh = zcr > 0.1 or sc > 2000
        flat_pitch = pitch_range < 20

        if loud and fast and harsh:
            return "angry"
        if loud and fast and not harsh:
            return "happy"
        if rms < 0.02 and (flat_pitch or tempo < 80):
            return "sad"
        if rms < 0.03 and harsh and not fast:
            return "anxious"
        return "neutral"

    except Exception as e:
        if VOICE_DEBUG:
            st.exception(e)
        return "neutral"

# =================== SUMMARIZATION ===================

def _normalize_spaces(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace("\r", " ").replace("\n", " ")

def split_sentences(txt: str):
    return re.split(r"(?<=[.!?])\s+", txt)

def summarize_text(text: str, n_sentences: int = 3) -> str:
    """Simple TF-IDF sentence ranking summarizer (no external API)."""
    if not text or len(text.split()) < 30:
        return text.strip()

    sentences = [s.strip() for s in split_sentences(text) if s.strip()]
    if len(sentences) <= n_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    scores = tfidf.sum(axis=1).A.ravel()
    top_idx = scores.argsort()[-n_sentences:][::-1]
    top_sentences = [sentences[i] for i in sorted(top_idx)]
    return " ".join(top_sentences)

# =================== CALENDAR EXPORT (ICS) ===================

def dt_to_ics(dt_obj: dt.datetime) -> str:
    return dt_obj.strftime("%Y%m%dT%H%M%S")

def _sanitize_ics_text(s: str) -> str:
    return _normalize_spaces(s).replace(",", r"\,").replace(";", r"\;")

def generate_ics_from_logs(df: pd.DataFrame, default_hour: int = 21) -> str:
    """Create ICS file where each entry is an event at default_hour local time."""
    header = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//MindSpace Companion//EN\n"
    body = ""
    now = dt.datetime.utcnow()
    for _, row in df.iterrows():
        try:
            d = pd.to_datetime(row["date"]).to_pydatetime()
            start = d.replace(hour=default_hour, minute=0, second=0)
            uid = f"{row['date']}-{row.get('time','00:00')}-{row['mood']}@mindspace"
            summary = f"Mood: {row['mood'].capitalize()} {EMOJI.get(row['mood'],'')}"
            desc = _sanitize_ics_text(str(row.get("text", "")))
            body += (
                "BEGIN:VEVENT\n"
                f"DTSTAMP:{dt_to_ics(now)}Z\n"
                f"DTSTART:{dt_to_ics(start)}\n"
                f"SUMMARY:{summary}\n"
                f"UID:{uid}\n"
                f"DESCRIPTION:{desc}\n"
                "END:VEVENT\n"
            )
        except Exception:
            continue
    return header + body + "END:VCALENDAR\n"

def generate_daily_selfcare_ics(hour: int = 21, minute: int = 0) -> str:
    """Creates a recurring daily self-care reminder."""
    now = dt.datetime.utcnow()
    start_local = dt.datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    header = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//MindSpace Companion//EN\n"
    event = (
        "BEGIN:VEVENT\n"
        f"DTSTAMP:{dt_to_ics(now)}Z\n"
        f"DTSTART:{dt_to_ics(start_local)}\n"
        "RRULE:FREQ=DAILY\n"
        "SUMMARY:MindSpace self-care check-in\n"
        "UID:selfcare-daily@mindspace\n"
        "DESCRIPTION:Take 5 minutes to breathe, reflect, and log your mood.\n"
        "END:VEVENT\n"
    )
    return header + event + "END:VCALENDAR\n"

# =================== WEEKLY SUMMARY ===================

def generate_weekly_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data to summarize."
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    seven_days_ago = pd.to_datetime(dt.date.today() - timedelta(days=7))
    last_7 = df[df["date"] >= seven_days_ago]
    if last_7.empty:
        return "No entries in the last 7 days."
    counts = last_7["mood"].value_counts().to_dict()
    most_common = max(counts, key=counts.get)
    text_blob = " ".join(last_7.get("text", "").astype(str))
    summary = summarize_text(text_blob, n_sentences=4) if text_blob.strip() else ""
    lines = [
        f"**Weekly Mood Summary ({seven_days_ago.date()} → {dt.date.today()})**",
        f"- Total entries: {len(last_7)}",
        f"- Most frequent mood: **{most_common.capitalize()} {EMOJI.get(most_common, '')}** ({counts[most_common]} times)",
    ]
    if summary:
        lines.append("\n**What you wrote about the most:**")
        lines.append(summary)
    return "\n".join(lines)

# =================== UI HELPERS ===================

def mood_calendar():
    with st.sidebar:
        st.markdown("### 📆 Mood Calendar")
        today = dt.date.today()
        cal = calendar.Calendar()
        month_days = cal.monthdayscalendar(today.year, today.month)

        mood_data = {}
        for log in st.session_state.logs:
            try:
                log_date = dt.datetime.strptime(log["date"], "%Y-%m-%d").date()
                if log_date.month == today.month and log_date.year == today.year:
                    mood_data[log_date.day] = log["mood"]
            except Exception:
                continue

        for week in month_days:
            cols = st.columns(7)
            for i, day in enumerate(week):
                with cols[i]:
                    if day != 0:
                        emoji = EMOJI.get(mood_data.get(day, ""), "⬜")
                        bg = "#ffccdd" if day == today.day else "#f8f8f8"
                        st.markdown(
                            f"""
                        <div style=\"text-align:center; padding:4px; background-color:{bg}; border-radius:6px;\">
                            {day}<br>{emoji}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

def render_spotify_playlist(mood: str):
    playlist_id = SPOTIFY_PLAYLISTS.get(mood)
    if playlist_id:
        st.subheader("🎧 Spotify playlist for your mood")
        components.iframe(f"https://open.spotify.com/embed/playlist/{playlist_id}", height=352)
        st.link_button("Open in Spotify", f"https://open.spotify.com/playlist/{playlist_id}")

def show_results(mood: str, source_text: str):
    st.success(f"🧠 Detected Mood: **{mood.capitalize()} {EMOJI.get(mood, '')}**")
    st.subheader("💖 Daily Affirmation")
    st.markdown(f"*{random.choice(MOTIVATIONAL_QUOTES[mood])}*")

    st.subheader("🎭 Mood Booster")
    gif = MOOD_GIF_LINKS.get(mood, "")
    if gif:
        st.image(gif, width=400)

    render_spotify_playlist(mood)

    note = st.text_area("Add a journal note (optional)", key=f"note_{mood}")
    if st.button("Save Entry", key=f"save_{mood}"):
        entry = {
            "date": dt.date.today().isoformat(),
            "time": dt.datetime.now().strftime("%H:%M"),
            "mood": mood,
            "text": f"{source_text}\n\nNotes: {note}" if note else source_text,
        }
        st.session_state.logs.append(entry)
        with open(LOG_FILE, "w") as f:
            json.dump(st.session_state.logs, f)
        st.success("Saved to journal!")
        st.rerun()

def mood_analyzer():
    st.header("🔍 Mood Analyzer")
    tab1, tab2 = st.tabs(["Text", "Voice"])

    with tab1:
        text = st.text_area("Describe how you're feeling...", height=120)
        if st.button("Analyze Text", type="primary"):
            if text.strip():
                mood = detect_text_emotion(text)
                show_results(mood, text)
            else:
                st.warning("Please type something first.")

    with tab2:
        if not AUDIO_AVAILABLE:
            st.info("Voice analysis not available on this system (`sounddevice` not installed).")
            return
        if not LIBROSA_AVAILABLE:
            st.info("`librosa` not installed — `pip install librosa` for voice features.")
            return
        global VOICE_DEBUG
        VOICE_DEBUG = st.toggle("Show raw voice features (debug)", value=False, key="voice_debug")
        duration = st.slider("Record duration (sec)", 3, 10, 5)
        if st.button("Record Voice 🎙️"):
            with st.spinner("Recording..."):
                audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
                sd.wait()
                buffer = io.BytesIO()
                sf.write(buffer, audio, 44100, format="WAV")
                mood = analyze_voice(buffer.getvalue())
                show_results(mood, "Voice Input")

# =================== PDF UPLOADER ===================

def pdf_uploader():
    st.subheader("📄 Upload PDF journals (auto-detect mood)")
    if not PDF_AVAILABLE:
        st.info("PyPDF2 not installed — `pip install PyPDF2` to enable PDF uploads.")
        return
    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        updated = False
        for f in files:
            try:
                reader = PyPDF2.PdfReader(f)
                pages_text = []
                for page in reader.pages:
                    try:
                        pages_text.append(page.extract_text() or "")
                    except Exception:
                        pages_text.append("")
                full_text = "\n".join(pages_text)
                if full_text.strip():
                    mood = detect_text_emotion(full_text)
                    entry = {
                        "date": dt.date.today().isoformat(),
                        "time": dt.datetime.now().strftime("%H:%M"),
                        "mood": mood,
                        "text": f"[Imported from {f.name}]\n\n{full_text[:5000]}",
                    }
                    st.session_state.logs.append(entry)
                    st.success(f"Imported {f.name} → Detected mood: {mood}")
                    updated = True
                else:
                    st.warning(f"No extractable text in {f.name}.")
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
        if updated:
            with open(LOG_FILE, "w") as wf:
                json.dump(st.session_state.logs, wf)

# =================== JOURNAL ===================

def journal():
    st.header("📖 Journal")
    pdf_uploader()

    if not st.session_state.logs:
        st.info("No entries yet.")
        return

    # Sort latest first
    for i, entry in enumerate(sorted(st.session_state.logs, key=lambda x: (x.get("date",""), x.get("time","")), reverse=True)):
        with st.expander(f"{entry['date']} {entry['time']} — {entry['mood'].capitalize()} {EMOJI.get(entry['mood'], '')}"):
            st.write(entry.get("text", ""))
            if st.button("Delete", key=f"del_{i}"):
                st.session_state.logs.remove(entry)
                with open(LOG_FILE, "w") as f:
                    json.dump(st.session_state.logs, f)
                st.rerun()

    # Export tools
    st.markdown("---")
    st.subheader("⬇️ Export")
    df = pd.DataFrame(st.session_state.logs)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV of journal", csv_data, file_name="mood_journal.csv", mime="text/csv")

    default_hour = st.number_input("Hour for exported ICS events (0-23)", min_value=0, max_value=23, value=21)
    if st.button("Generate .ics from journal entries"):
        ics = generate_ics_from_logs(df, default_hour=int(default_hour))
        st.download_button("Download journal.ics", ics, "journal.ics", "text/calendar", key="dl_journal_ics")

    sc_hour = st.number_input("Daily self-care reminder hour (0-23)", min_value=0, max_value=23, value=21, key="sc_hour")
    sc_min = st.number_input("Daily self-care reminder minute (0-59)", min_value=0, max_value=59, value=0, key="sc_min")
    if st.button("Generate daily self-care .ics"):
        ics2 = generate_daily_selfcare_ics(hour=int(sc_hour), minute=int(sc_min))
        st.download_button("Download selfcare.ics", ics2, "selfcare.ics", "text/calendar", key="dl_selfcare_ics")

# =================== ANALYTICS ===================

@st.cache_data(show_spinner=False)
def _df_from_logs(logs):
    if not logs:
        return pd.DataFrame(columns=["date","time","mood","text"])
    df = pd.DataFrame(logs)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def analytics():
    st.header("📊 Mood Analytics")
    if not st.session_state.logs:
        st.warning("No data yet.")
        return

    df = _df_from_logs(st.session_state.logs)
    if df.empty or "date" not in df.columns:
        st.error("Log entries missing 'date' field.")
        return

    # Mood distribution
    st.subheader("Your Mood Patterns")
    mood_counts = df["mood"].value_counts().reset_index()
    mood_counts.columns = ["Mood", "Count"]
    chart = alt.Chart(mood_counts).mark_bar().encode(
        x=alt.X("Mood:N", sort=None),
        y=alt.Y("Count:Q"),
        color=alt.Color("Mood:N")
    )
    st.altair_chart(chart, use_container_width=True)

    # Timeline
    st.subheader("Mood Timeline")
    mood_order = list(EMOJI.keys())
    df["mood_cat"] = pd.Categorical(df["mood"], categories=mood_order, ordered=True)
    timeline = (
        alt.Chart(df.dropna(subset=["date"]))
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("mood_cat:N", sort=mood_order, title="Mood"),
            color=alt.Color("mood:N"),
            tooltip=["date:T", "mood:N"]
        )
    )
    st.altair_chart(timeline, use_container_width=True)

    # Word cloud
    st.subheader("Common Themes")
    if "text" in df.columns and df["text"].notna().any():
        text_blob = " ".join(df["text"].dropna().astype(str))
        if text_blob.strip():
            wordcloud = WordCloud(width=800, height=400).generate(text_blob)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No text to analyze.")
    else:
        st.info("No text to analyze.")

    # Weekly AI-ish summary
    st.subheader("🧾 Weekly AI Summary")
    st.markdown(generate_weekly_summary(df))

# =================== MAIN ===================

def main():
    st.title("📘 MindSpace Companion")
    st.subheader("Reflect. Record. Rediscover Yourself.")

    mood_calendar()
    tab1, tab2, tab3 = st.tabs(["Analyzer", "Journal", "Analytics"])
    with tab1:
        mood_analyzer()
    with tab2:
        journal()
    with tab3:
        analytics()
    st.sidebar.markdown("---")
    st.sidebar.info("Take care of your mind 💖")

if __name__ == "__main__":
    main()

