# CMP7005 PRAC1 — From Data to Application Development

**Module:** CMP7005 Programming for Data Analysis  
**Academic Year:** 2025–26 | **Semester:** 2  
**Student ID:** [Your Student ID]

---

## Project Overview

This project analyses hourly air quality data from four Beijing monitoring stations (2013–2017) to:
- Perform comprehensive Exploratory Data Analysis (EDA)
- Build a Random Forest model to predict PM2.5 concentrations
- Deliver an interactive Streamlit dashboard for data exploration and forecasting

## Selected Stations

| Station | Type | District |
|---|---|---|
| Dongsi | Urban | Dongcheng |
| Wanshouxigong | Urban | Xuanwu |
| Dingling | Suburban | Changping |
| Huairou | Suburban | Huairou |

## File Structure

```
CMP7005-PRAC1/
├── CMP7005_PRAC1.ipynb          ← Main notebook (Tasks 1–5)
├── app.py                        ← Streamlit GUI application (Task 4)
├── build_notebook.py             ← Script that generated the notebook
├── requirements.txt              ← Python dependencies
├── beijing_air_quality_processed.csv  ← Generated after running notebook
├── rf_pm25_model.pkl             ← Trained model (generated after Task 3)
└── README.md
```

## Setup Instructions (Google Colab)

1. Upload `CMP7005_PRAC1.ipynb` to Google Colab
2. Run all cells top to bottom — the dataset downloads automatically
3. For the Streamlit app, follow the instructions in the **Task 4** cell

## Setup Instructions (Local)

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/CMP7005-PRAC1.git
cd CMP7005-PRAC1

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook CMP7005_PRAC1.ipynb

# Run Streamlit app (after running notebook)
streamlit run app.py
```

## Requirements

See `requirements.txt` for full list. Key libraries:
- pandas, numpy, matplotlib, seaborn, plotly
- scikit-learn, scipy, joblib
- streamlit

## Dataset

Beijing Multi-Site Air Quality Data  
Source: UCI ML Repository (Dataset ID 501)  
URL: https://archive.ics.uci.edu/dataset/501/beijing+multi-site+air-quality+data  
Period: 1 March 2013 – 28 February 2017

## AI Usage Statement

AI tools (Claude) were used to assist with initial code structure suggestions and debugging guidance. All code has been reviewed, understood, tested, and adapted by the student. Any AI-assisted content has been rewritten and integrated in accordance with Cardiff Met's AI Acknowledged policy.
