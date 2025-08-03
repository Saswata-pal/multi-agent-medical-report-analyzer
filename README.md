# 🩺 AI-Powered Medical Report Analyzer

This application leverages **Gemini AI**, **CrewAI agents**, and **OCR** to analyze medical reports (PDFs or images). It extracts test results, checks for abnormalities, visualizes the data, and provides a simple summary and conclusion.

## 🚀 Features

- 🧠 Multi-agent pipeline (OCR → Parsing → Analysis → Summary)
- 🔍 Gemini AI-based understanding of test values
- 📊 Interactive visualizations (Bar + Pie charts)
- 🌐 Multi-language translation
- 💾 Downloadable results (summary, JSON, report)

## 🛠️ Built With

- [Streamlit](https://streamlit.io/)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [Gemini API](https://ai.google.dev/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Plotly](https://plotly.com/python/)
- [Google Translator](https://pypi.org/project/deep-translator/)

---

## 📸 Demo

![Demo Screenshot](demo.png) *(Replace with a real screenshot)*

---

## 🔧 Installation

```bash
git clone https://github.com/Saswata-pal/multi-agent-medical-report-analyzer
cd multi-agent-medical-report-analyzer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
