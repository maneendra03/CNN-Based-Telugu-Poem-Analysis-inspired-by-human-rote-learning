# తెలుగు కవిత్వ AI (Telugu Poetry AI)
## CNN-Based Poem Learning & Interpretation Inspired by Human Rote Learning

---

## 🎯 ప్రాజెక్ట్ అవలోకనం (Project Overview)

ఈ ప్రాజెక్ట్ **తెలుగు కవిత్వం** కోసం ప్రత్యేకంగా రూపొందించబడిన AI వ్యవస్థ. మానవులు పునరావృత అభ్యాసం ద్వారా కవిత్వాన్ని ఎలా నేర్చుకుంటారో అదే విధంగా ఈ వ్యవస్థ పని చేస్తుంది.

This project is a novel AI system specifically designed for **Telugu poetry**. It mimics how humans learn poetry through repetition (rote learning).

---

## 📁 ప్రాజెక్ట్ నిర్మాణం (Project Structure)

```
majorproject - A/
├── app/
│   ├── ui.py                    # English UI
│   └── telugu_ui.py             # తెలుగు UI
├── config/
│   └── config.yaml
├── data/
│   ├── knowledge_base/
│   │   ├── telugu_prosody.json  # తెలుగు ఛందస్సు నియమాలు
│   │   ├── poetic_rules.json
│   │   └── grammar_rules.json
│   └── processed/
│       ├── telugu_poems.json    # తెలుగు కవితలు
│       └── ...
├── scripts/
│   ├── train_telugu.py          # తెలుగు శిక్షణ
│   ├── download_telugu_datasets.py
│   └── ...
├── src/
│   ├── models/
│   │   ├── telugu_backbone.py   # IndicBERT/MuRIL
│   │   ├── cnn_module.py
│   │   ├── memory_attention.py  # రైట్ లెర్నింగ్ మెమరీ
│   │   └── ...
│   ├── preprocessing/
│   │   ├── telugu_cleaner.py    # తెలుగు టెక్స్ట్ శుద్ధి
│   │   └── ...
│   └── evaluation/
│       ├── telugu_metrics.py    # తెలుగు మెట్రిక్స్
│       └── ...
└── README.md
```

---

## 🧠 తెలుగు మోడల్ వివరాలు (Telugu Model Details)

### Pre-trained Models (పూర్వ శిక్షణ పొందిన మోడల్స్)

| Model | Parameters | Telugu Support |
|-------|------------|----------------|
| **IndicBERT** | 110M | ✅ Excellent |
| **MuRIL** | 237M | ✅ Very Good |
| **mBERT** | 110M | ✅ Good |

### ఎందుకు IndicBERT/MuRIL?

| English GPT-2 | IndicBERT/MuRIL |
|---------------|-----------------|
| ఆంగ్లం మాత్రమే | 12+ భారతీయ భాషలు |
| తెలుగు లేదు | తెలుగు బాగా మద్దతు |
| 50K vocab | Telugu tokens included |

---

## 📜 తెలుగు ఛందస్సు (Telugu Prosody)

### ప్రాస (Praasa - Rhyme)
```
ఉప్పు కప్పురంబు నొక్కపోలికనుండు   ← 'ప్పు'
చూడ చూడ రుచులు జాడ వేరు          ← 'డ'
```
**ప్రాస = రెండవ అక్షరం ఒకే విధంగా ఉండాలి** (Second syllable must match)

### ఛందస్సు రకాలు (Meter Types)

| ఛందస్సు | అక్షరాలు/పంక్తి |
|---------|-----------------|
| ఉత్పలమాల | 20 |
| చంపకమాల | 21 |
| కందం | 8-12 |
| ఆటవెలది | 8-10 |
| తేటగీతి | 6-8 |

### గణాలు (Ganaalu - Metrical Feet)

| గణం | Pattern | Example |
|-----|---------|---------|
| య | లఘు-గురు-గురు | I U U |
| మ | గురు-గురు-గురు | U U U |
| త | గురు-గురు-లఘు | U U I |
| భ | గురు-లఘు-లఘు | U I I |

---

## 🏃 ఎలా నడపాలి (How to Run)

### 1. డేటాసెట్ సృష్టించు (Create Dataset)
```bash
python scripts/download_telugu_datasets.py
```

### 2. శిక్షణ (Training)
```bash
python scripts/train_telugu.py
```

### 3. UI ప్రారంభించు (Launch UI)
```bash
python app/telugu_ui.py
# Open http://localhost:7861
```

---

## 📊 మూల్యాంకన మెట్రిక్స్ (Evaluation Metrics)

| Metric | Description |
|--------|-------------|
| **ప్రాస ఖచ్చితత్వం** | Praasa (rhyme) accuracy |
| **ఛందస్సు స్థిరత్వం** | Meter consistency |
| **తెలుగు నిష్పత్తి** | Telugu character ratio |
| **పదజాల వైవిధ్యం** | Vocabulary diversity |

---

## 📚 డేటాసెట్ వనరులు (Dataset Sources)

### చేర్చబడిన కవితలు (Included Poems)

| కవి | కవితల సంఖ్య | శైలి |
|-----|-------------|------|
| వేమన | 5+ | ఆట వెలది |
| సుమతీ శతకం | 3+ | కందం |
| అన్నమయ్య | 3+ | సంకీర్తన |
| జానపద | 4+ | గేయం |
| పోతన | 2+ | ఉత్పలమాల |
| ఆధునిక | 4+ | వచన కవిత |

### బాహ్య వనరులు (External Sources)

- [Telugu Wikisource](https://te.wikisource.org/)
- [Project Madurai](https://www.projectmadurai.org/)
- [Kaggle Telugu NLP](https://www.kaggle.com/datasets/sudalairajkumar/telugu-nlp)

---

## ⭐ ప్రత్యేకత (Uniqueness)

| సాధారణ విధానం | మా విధానం |
|----------------|-----------|
| ఆంగ్ల GPT-2 | **తెలుగు IndicBERT** |
| సాధారణ అటెన్షన్ | **రైట్ లెర్నింగ్ మెమరీ** |
| ఒకసారి ఉత్పత్తి | **ఇటరేటివ్ ఫీడ్‌బ్యాక్ లూప్** |
| BLEU మాత్రమే | **ప్రాస + ఛందస్సు మెట్రిక్స్** |

---

## 📞 సంప్రదించండి (Contact)

**Author**: Maneendra  
**GitHub**: [maneendra03](https://github.com/maneendra03)  
**Project**: [CNN-Based-Poem-Learning](https://github.com/maneendra03/CNN-Based-Poem-Learning-Interpretation-Inspired-by-Human-Rote-Learning-)

---

*తెలుగు భాష వర్ధిల్లాలి! 🙏*
