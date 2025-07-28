
# 🧠 MindSpace Companion

**An AI-powered mental health journaling app built with Streamlit**  
Track your mood, reflect through voice or text, and get personalized support every day.

---

## ✨ Features

- **💬 Text Emotion Detection** using VADER Sentiment Analysis
- **🎙️ Voice-based Emotion Detection** using Librosa & SoundDevice
- **📖 Journaling with Auto Mood Detection**
- **📄 PDF Journal Uploads**
- **📅 Mood Calendar** with emoji visualization
- **📊 Analytics Dashboard** (mood patterns, word clouds, trends)
- **📝 Weekly AI Summary** using TF-IDF-based summarization
- **🎧 Spotify Mood Playlists** to boost your emotional state
- **📥 Export Options:** CSV + ICS (calendar) support
- **💡 Self-Care Suggestions** tailored to your mood

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **NLP**: VADER, scikit-learn (TF-IDF)
- **Audio Analysis**: Librosa, SoundDevice
- **Visualization**: Altair, Matplotlib, WordCloud
- **File Parsing**: PyPDF2
- **Data Storage**: Local JSON (`mood_log.json`)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mindspace-companion.git
cd mindspace-companion

2. Install Requirements
pip install -r requirements.txt

3. Run the App
streamlit run app.py


