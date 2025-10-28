# 🧠 AI YouTube Trailer Generator

An intelligent system that automatically creates **20-second highlight trailers** from full YouTube videos.  
It uses **YouTube API**, **OpenCV**, **Librosa**, and a **trainable AI model** to detect visually and audibly exciting moments, guided by **user preferences** (like *action*, *funny*, *informative*, etc.).

---

## 🎯 Key Features

- 🎬 Automatically downloads videos from YouTube (via `yt_dlp`)
- 🧩 Extracts multi-modal features:
  - Motion dynamics (OpenCV histogram difference)
  - Audio energy (Librosa RMS)
  - Text/Subtitle cues (caption density)
- 🧠 Trainable highlight model (`LogisticRegression` / `LightGBM`)
- 🎛️ Interactive user preferences: *style*, *focus*, *category*, *pace*
- ⚡ Creates short, cinematic trailers (default: 20 seconds)
- 📊 Includes evaluation metrics (Precision@K, NDCG, MAP)
- 💾 Supports saving features, models, and generated trailers

---

## 🗂️ Project Structure

```
trailer_ai_package/
│
├── data/
│   ├── raw/           # downloaded YouTube videos (.mp4)
│   ├── features/      # extracted per-chunk feature CSVs
│   ├── models/        # trained highlight models (.pkl)
│   ├── out/           # generated trailers
│   └── video_ids.csv  # list of YouTube video IDs
│
├── trailer_ai/
│   ├── io_utils.py
│   ├── feature_extraction.py
│   ├── model_train.py
│   ├── trailer_render.py
│   └── preferences.py
│
├── TrailerAI.ipynb
├── .env
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install yt-dlp opencv-python librosa numpy pandas tqdm scikit-learn joblib python-dotenv
```

### 2. Install ffmpeg
**macOS**
```bash
brew install ffmpeg
```
**Windows**
```bash
choco install ffmpeg
```
**Ubuntu**
```bash
sudo apt install ffmpeg
```

### 3. Add your YouTube API key  
Create a file named `.env`:
```bash
TRAILER_API_TOKEN=YOUR_YOUTUBE_API_KEY
TRAILER_WRITE_SUBS=0
```

---

## 🚀 How to Run (Notebook Mode)

1. Open `TrailerAI.ipynb`
2. Run setup cells (imports + functions)
3. Run the training cell if needed  
   → Creates `data/models/highlight_model.pkl`
4. Run the interactive trailer cell:
   - Choose **1** to paste a YouTube URL  
   - Choose **2** to pick from your CSV  
   - Enter your preferences
5. Trailer saved in `data/out/VIDEOID_trailer.mp4`

---

## 🧮 Model Training & Evaluation

- Features: `[motion, audio, caption_overlap, keyword_density]`
- Model: `LogisticRegression`
- Metrics: Precision@K, Recall@K, F1@K, NDCG@K, MAP

---

## 🎚️ User Preference Parameters

| Parameter | Options | Description |
|------------|----------|--------------|
| `style` | action, emotional, funny, informative | Affects weighting of motion/audio/text |
| `focus` | dialogue, visuals, both | Bias toward spoken content or visual motion |
| `category` | sports, nature, music, etc. | Tweaks scene detection thresholds |
| `pace` | fast, medium, slow | Controls average clip length |
| `blend α` | 0–1 | 0 = user prefs only, 1 = model only |

---

## 🧹 Cleanup Tip

To free up space safely:
```bash
rm -rf data/raw
```

---

## 💡 Future Enhancements

- Transformer-based highlight detection (CLIP / VideoMAE)
- Automatic background music
- Multilingual subtitle emotion detection
- Streamlit web UI

---

## 👨‍💻 Author

**Dhruv Patel**  
MSc Computer Science, Lakehead University  
📧 dhruvpatel.work2024@gmail.com
🔗 GitHub: [DhruvPatel](https://github.com/Dhruv1227)
