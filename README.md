# 🧠 Health Symptom Mapping to Diagnosis Using Patient Reviews  

### 🔍 Overview  
This project uses **Natural Language Processing (NLP)** and **Machine Learning** to map patient-written reviews or informal symptom descriptions into standardized medical diagnoses.  
It bridges the gap between **everyday language** (used by patients) and **clinical terminology**, improving understanding and supporting healthcare professionals in early-stage diagnosis.

---

### 💡 Motivation  
Patients often describe symptoms online or during consultations in casual, non-medical language (e.g., *“I feel dizzy and have stomach cramps”*).  
Traditional medical systems use structured terminologies (like ICD-10 or SNOMED CT), leading to interpretation mismatches.  
This project develops an intelligent system to translate such informal text into accurate, standardized diagnoses.

---

### 🧱 System Architecture  

**Input:** Raw patient review text  
**Process:**  
1. **Data Preprocessing:** Cleans text, removes noise, tokenizes, and lemmatizes.  
2. **Symptom Extraction:** Identifies relevant symptoms using regex and keyword heuristics (extendable to NER models).  
3. **Symptom Mapping:** Maps symptoms to standardized diagnoses using TF-IDF + semantic similarity.  
4. **Classification (optional):** Uses an LSTM or traditional ML models for disease prediction.  
5. **Output:** Candidate diagnoses with confidence scores.

**Output Example:**  
Read 5 reviews
------------------------------------------------------------
Review 1: I have been having a persistent cough and sore throat for 3 days.
Extracted symptom phrases: ['persistent cough', 'sore throat']
Candidate diagnoses: [('Common Cold', 0.65)]

Review 2: Stomach cramps and nausea since yesterday.
Extracted symptom phrases: ['Stomach cramps', 'nausea since yesterday', 'nausea']
Candidate diagnoses: [('Gastroenteritis', 0.60)]


health_symptom_mapping/
│
├── src/
│ ├── data_preprocessing.py
│ ├── symptom_extraction.py
│ ├── symptom_mapping.py
│ ├── classification.py
│ └── main.py
│
├── data/
│ └── sample_reviews.txt
│
├── results/
│ └── predictions.txt
│
├── notebooks/
│ └── LSTM_Health_Symptom.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md


---

### 🚀 Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
**Create and activate a virtual environment**:
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
**Install dependencies**:
pip install -r requirements.txt
**Run the pipeline**:
python -m src.main --reviews data/sample_reviews.txt
