# app.py
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
import chromadb.utils.embedding_functions as ef
ef.DefaultEmbeddingFunction = lambda: ef.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("GEMINI_API_KEY"))

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

from PIL import Image
import pdfplumber
import streamlit as st
import re
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Type
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

# === TOOLS ===
from deep_translator import GoogleTranslator

class TranslationInput(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(default="es", description="Target language code (e.g., 'es' for Spanish)")

class TranslationTool(BaseTool):
    name: str = "Translation Tool"
    description: str = "Translates text to specified language using Google Translate"
    args_schema: Type[BaseModel] = TranslationInput

    def _run(self, text: str, target_language: str = "es") -> str:
        try:
            translator = GoogleTranslator(source='en', target=target_language)
            result = translator.translate(text)
            return result
        except Exception as e:
            return f"Translation error: {str(e)}"
        


class OCRInput(BaseModel):
    file_path: str = Field(..., description="Path to uploaded PDF or image")

class OCRTool(BaseTool):
    name: str = "OCR Tool"
    description: str = "Extracts text from a medical PDF or image file"
    args_schema: Type[BaseModel] = OCRInput

    def _run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: File not found at path: {file_path}"
        
        try:
            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            else:
                img = Image.open(file_path)
                return pytesseract.image_to_string(img)
        except Exception as e:
            return f"Error processing file: {str(e)}"

class ParserInput(BaseModel):
    text: str = Field(..., description="Raw text extracted from the report")

class ParserTool(BaseTool):
    name: str = "Parser Tool"
    description: str = "Extracts medical test names and values from raw text using enhanced parsing"
    args_schema: Type[BaseModel] = ParserInput

    def _run(self, text: str) -> str:
        # Enhanced prompt for Gemini to parse medical data
        prompt = f"""Extract ALL medical test names and their values from this medical report text. 

Text: {text}

Instructions:
1. Find every test name and its corresponding value
2. Include full test names (don't truncate)
3. Handle various formats like "Test Name: Value", "Test Name Value", "Test Name - Value"
4. Include units if present but focus on the numeric value
5. Return ONLY valid JSON in this exact format:

{{"test_name": "value"}}

Examples:
- "Hemoglobin 15.2 g/dL" → {{"Hemoglobin": "15.2"}}
- "Total Leucocyte Count: 5000 cells/μL" → {{"Total Leucocyte Count": "5000"}}
- "Neutrophils (%) 50" → {{"Neutrophils": "50"}}

Extract ALL tests found in the report."""

        # Call Gemini API for parsing
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            gemini_response = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean and extract JSON
            cleaned = re.sub(r'```json\s*|\s*```|```', '', gemini_response).strip()
            
            try:
                parsed_json = json.loads(cleaned)
                print(f"DEBUG - Parsed tests: {len(parsed_json)} tests found")
                return json.dumps(parsed_json)
            except:
                print("DEBUG - Gemini JSON parsing failed, using fallback")
                return self._fallback_parsing(text)
                
        except Exception as e:
            print(f"Gemini parsing error: {str(e)}")
            return self._fallback_parsing(text)

    def _fallback_parsing(self, text: str) -> str:
        """Enhanced fallback parsing with multiple regex patterns"""
        results = {}
        lines = text.split("\n")
        
        # Multiple regex patterns for different medical report formats
        patterns = [
            r"([A-Za-z][A-Za-z0-9\s\(\)\-\/]+?)\s*[:]\s*(\d+\.?\d*)",  # "Test Name: Value"
            r"([A-Za-z][A-Za-z0-9\s\(\)\-\/]+?)\s+(\d+\.?\d*)\s*[a-zA-Z\/]*",  # "Test Name Value unit"
            r"([A-Za-z][A-Za-z0-9\s\(\)\-\/]+?)\s*[-]\s*(\d+\.?\d*)",  # "Test Name - Value"
            r"([A-Za-z][A-Za-z0-9\s\(\)\-\/]+?)\s*[=]\s*(\d+\.?\d*)",  # "Test Name = Value"
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    test_name = match.group(1).strip()
                    value = match.group(2).strip()
                    
                    #
                    test_name = re.sub(r'\s+', ' ', test_name)  
                    test_name = test_name.strip('()[]{}')  
                    
                    if len(test_name) > 3 and test_name not in results: 
                        results[test_name] = value
                        break
        
        print(f"DEBUG - Fallback parsing found: {len(results)} tests")
        return json.dumps(results)

class AnalyzerInput(BaseModel):
    results: str = Field(..., description="JSON string of parsed test results")


class AnalyzerTool(BaseTool):
    name: str = "Analyzer Tool"
    description: str = "Analyzes medical test values for abnormal ranges using Gemini API"
    args_schema: Type[BaseModel] = AnalyzerInput

    def _run(self, results: str) -> str:
        try:
            tests = json.loads(results)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        
        # Enhanced prompt for any medical test type
        prompt = f"""Analyze these medical test results and return ONLY a valid JSON object:

Test Data: {json.dumps(tests)}

Return analysis in this EXACT format (no markdown, no explanations):
{{
    "test_name": {{"value": number, "status": "Normal/High/Low/Unknown"}}
}}

Analyze based on standard medical ranges for:
- Blood tests (CBC, chemistry panels)
- Urine tests
- Cardiac markers
- Liver/kidney function
- Thyroid tests
- Any other medical tests

Example output:
{{"Hemoglobin": {{"value": 15, "status": "Normal"}}, "Total Leucocyte Count": {{"value": 5000, "status": "Normal"}}}}
"""

        # Call Gemini API
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            gemini_response = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean response and extract JSON
            cleaned = re.sub(r'```json\s*|\s*```|```', '', gemini_response).strip()
            
            # to parse as JSON
            try:
                parsed_json = json.loads(cleaned)
                return json.dumps(parsed_json)
            except:
                # Extract JSON pattern if embedded in text
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
                if json_match:
                    return json_match.group()
                else:
                    return self._create_fallback_analysis(tests)
                
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
            return self._create_fallback_analysis(tests)


    def _create_fallback_analysis(self, tests):
        """Fallback analysis with expanded reference ranges"""
        reference_ranges = {
            # Blood Count
            "Hemoglobin": (12.0, 17.5), "Haemoglobin": (12.0, 17.5),
            "Total Leucocyte Count": (4000, 11000), "WBC": (4000, 11000),
            "Neutrophils": (40, 70), "Lymphocytes": (20, 45),
            "Eosinophils": (1, 4), "Monocytes": (2, 10), "Basophils": (0, 2),
            "Absolute Neutrophils": (1500, 8000), "Absolute Lymphocytes": (1000, 4000),
            "Absolute Eosinophils": (15, 500), "Absolute Monocytes": (200, 1000),
            "RBC Count": (4.2, 5.4), "Hematocrit": (36, 46), "MCV": (80, 100),
            "MCH": (27, 32), "MCHC": (32, 36), "Platelet Count": (150000, 450000),
            
            # Chemistry Panel
            "Glucose": (70, 100), "Creatinine": (0.6, 1.2), "BUN": (7, 20),
            "Sodium": (136, 145), "Potassium": (3.5, 5.1), "Chloride": (98, 107),
            "Total Protein": (6.0, 8.3), "Albumin": (3.5, 5.0),
            
            # Liver Function
            "ALT": (7, 56), "AST": (10, 40), "Bilirubin": (0.1, 1.2),
            "Alkaline Phosphatase": (44, 147),
            
            # Lipid Panel
            "Total Cholesterol": (125, 200), "HDL": (40, 60), "LDL": (0, 100),
            "Triglycerides": (0, 150),
            
            # Thyroid
            "TSH": (0.27, 4.2), "T3": (80, 200), "T4": (5.1, 14.1),
        }
        
        analyzed = {}
        for test, val in tests.items():
            try:
                if val is None or val == "":
                    analyzed[test] = {"value": "N/A", "status": "Unknown"}
                    continue

                val_num = float(val)
                # Find matching reference range (case-insensitive)
                range_found = None
                for ref_test, ref_range in reference_ranges.items():
                    if test.lower() == ref_test.lower() or ref_test.lower() in test.lower():
                        range_found = ref_range
                        break
                
                if range_found:
                    low, high = range_found
                    if val_num < low:
                        status = "Low"
                    elif val_num > high:
                        status = "High"
                    else:
                        status = "Normal"
                else:
                    status = "Unknown"
                    
                analyzed[test] = {"value": val_num, "status": status}
            except (ValueError, TypeError):
                analyzed[test] = {"value": str(val), "status": "Unknown"}
    
        return json.dumps(analyzed)


class SummaryInput(BaseModel):
    analysis: str = Field(..., description="JSON string of analyzed medical test results")
    target_language: str = Field(default="en", description="Target language code for translation")

class SummaryTool(BaseTool):
    name: str = "Summary Tool"
    description: str = "Summarizes analyzed results in user-friendly format using Gemini API"
    args_schema: Type[BaseModel] = SummaryInput

    def _run(self, analysis: str, target_language: str = "en") -> str: 
        try:
            data = json.loads(analysis)
        except json.JSONDecodeError as e:
            return f"Error: Unable to parse analysis data - {str(e)}"

        if "error" in data:
            return f"Analysis Error: {data['error']}"

        # Format analysis into readable input
        structured_summary = ""
        for test, info in data.items():
            val = info.get("value", "N/A")
            status = info.get("status", "Unknown")
            structured_summary += f"{test}: {val} ({status})\n"

        # Prompt for Gemini
        prompt = f"""You are a medical assistant helping patients understand their test reports.
Write a friendly, easy-to-read summary in bullet points using this data:

{structured_summary}

Avoid medical jargon. Explain if values are normal or abnormal and gently suggest next steps.
"""

        # Call Gemini API
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            summary = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # TRANSLATION 
            if target_language != "en":
                try:
                    translator = GoogleTranslator(source='en', target=target_language)
                    summary = translator.translate(summary)
                except Exception as e:
                    summary += f"\n\n(Translation to {target_language} failed: {str(e)})"
            
            return summary
            
        except Exception as e:
            return f"❌ Gemini API Error: {str(e)}"

# === VISUALIZATION FUNCTIONS ===

def create_test_results_chart(analysis_data):
    """Create interactive charts for test results"""
    try:
        if not analysis_data or analysis_data.strip() == "":
            st.warning("No analysis data available for visualization")
            return None
            
        data = json.loads(analysis_data)
        
        if "error" in data:
            st.error(f"Analysis error: {data['error']}")
            return None
            
        # Prepare data for visualization
        tests = []
        values = []
        statuses = []
        colors = []
        
        for test, info in data.items():
             if info.get("status") != "Invalid" and info.get("value") is not None:
                try:
                    # Ensure value can be converted to float
                    value = float(info.get("value", 0))
                    tests.append(test)
                    values.append(value)
                    status = info.get("status", "Unknown")
                    statuses.append(status)
                    
                    # Color coding
                    if status == "Normal":
                        colors.append("#28a745")  # Green
                    elif status == "High":
                        colors.append("#dc3545")  # Red
                    elif status == "Low":
                        colors.append("#ffc107")  # Yellow
                    else:
                        colors.append("#6c757d")  # Gray
                except (ValueError, TypeError) as e:
                    print(f"DEBUG - Skipping {test} due to invalid value: {info.get('value')} - {e}")
                    continue
        
        if not tests:
            st.warning("No valid test data found for visualization")
            return None
            
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=tests,
                y=values,
                marker_color=colors,
                text=statuses,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Value: %{y}<br>Status: %{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Medical Test Results Overview",
            xaxis_title="Test Name",
            yaxis_title="Test Value",
            showlegend=False,
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
        
    except json.JSONDecodeError as e:
        st.error(f"Error parsing analysis data: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        print(f"DEBUG - Chart creation error: {e}")
        return None

def create_status_pie_chart(analysis_data):
    """Create pie chart showing distribution of test statuses"""
    try:
        data = json.loads(analysis_data)
        if "error" in data:
            return None
            
        status_counts = {"Normal": 0, "High": 0, "Low": 0, "Unknown": 0}
        
        for test, info in data.items():
            status = info.get("status", "Unknown")
            if status in status_counts:
                status_counts[status] += 1
        
        # Filter out zero counts
        filtered_counts = {k: v for k, v in status_counts.items() if v > 0}
        
        if not filtered_counts:
            return None
            
        colors = {
            "Normal": "#28a745",
            "High": "#dc3545", 
            "Low": "#ffc107",
            "Unknown": "#6c757d"
        }
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(filtered_counts.keys()),
                values=list(filtered_counts.values()),
                marker_colors=[colors[k] for k in filtered_counts.keys()],
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Test Results Status Distribution",
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating pie chart: {str(e)}")
        return None

def generate_conclusion(analysis_data, summary_text):
    """Generate a comprehensive conclusion based on analysis"""
    try:
        data = json.loads(analysis_data)
        if "error" in data:
            return "Unable to generate conclusion due to analysis errors."
        
        total_tests = len(data)
        normal_count = sum(1 for info in data.values() if info.get("status") == "Normal")
        abnormal_count = total_tests - normal_count
        
        # Calculate percentages
        normal_percentage = (normal_count / total_tests * 100) if total_tests > 0 else 0
        abnormal_percentage = (abnormal_count / total_tests * 100) if total_tests > 0 else 0
        
        # Generate conclusion
        conclusion = f"""
## 📋 Medical Report Analysis Conclusion

### Summary Statistics:
- **Total Tests Analyzed:** {total_tests}
- **Normal Results:** {normal_count} ({normal_percentage:.1f}%)
- **Abnormal Results:** {abnormal_count} ({abnormal_percentage:.1f}%)

### Overall Assessment:
"""
        
        if abnormal_percentage == 0:
            conclusion += "🟢 **Excellent:** All test results are within normal ranges. This indicates good health status."
        elif abnormal_percentage <= 25:
            conclusion += "🟡 **Good:** Most test results are normal with minor abnormalities that may need attention."
        elif abnormal_percentage <= 50:
            conclusion += "🟠 **Moderate:** Several test results are outside normal ranges. Medical consultation recommended."
        else:
            conclusion += "🔴 **Attention Required:** Multiple abnormal results detected. Please consult with your healthcare provider promptly."
        
        conclusion += f"""

### Key Findings:
"""
        
        # Add specific findings
        high_values = [test for test, info in data.items() if info.get("status") == "High"]
        low_values = [test for test, info in data.items() if info.get("status") == "Low"]
        
        if high_values:
            conclusion += f"- **Elevated levels:** {', '.join(high_values)}\n"
        if low_values:
            conclusion += f"- **Below normal levels:** {', '.join(low_values)}\n"
        
        conclusion += """
### Recommendations:
1. **Share these results** with your healthcare provider
2. **Follow up** on any abnormal values as recommended
3. **Maintain regular** health check-ups
4. **Keep a record** of your test results for future reference

### Next Steps:
- Schedule a consultation with your doctor to discuss these results
- Ask about any lifestyle changes that might help improve abnormal values
- Consider retesting abnormal values as recommended by your healthcare provider

---
*This analysis is for informational purposes only and should not replace professional medical advice.*
        """
        
        return conclusion
        
    except Exception as e:
        return f"Error generating conclusion: {str(e)}"

# === STREAMLIT APP ===

st.set_page_config(page_title="🧠 Medical Report Analyzer", layout="wide")
st.title("🩺 AI-Powered Medical Report Analyzer")
st.markdown("Upload your medical report and get instant analysis with visualizations and insights!")

uploaded_file = st.file_uploader("📤 Upload Medical Report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Add this before the analyze button
    col1, col2 = st.columns(2)
    with col1:
        translate_option = st.checkbox("🌐 Translate Results")
    with col2:
        if translate_option:
            target_language = st.selectbox(
                "Select Language:",
                options=[
                    ("es", "Spanish"), ("fr", "French"), ("de", "German"),
                    ("it", "Italian"), ("pt", "Portuguese"), ("zh", "Chinese"),
                    ("ja", "Japanese"), ("ko", "Korean"), ("ar", "Arabic"),
                    ("hi", "Hindi"), ("ru", "Russian"), ("bn", "Bengali")
                ],
                format_func=lambda x: x[1]
            )
            target_lang_code = target_language[0]
        else:
            target_lang_code = "en"
    #Analyze button         
    if st.button("🧪 Analyze Report", type="primary"):
        with st.spinner("🔍 Running multi-agent analysis..."):

            # Define tools
            ocr_tool = OCRTool()
            parser_tool = ParserTool()
            analyzer_tool = AnalyzerTool()
            summary_tool = SummaryTool()

            # Define agents with Gemini LLM
            ocr_agent = Agent(
                role="OCR Specialist",
                goal="Extract raw medical text from uploaded reports.",
                backstory="Expert in OCR technology and document parsing.",
                tools=[ocr_tool],
                allow_delegation=False,
                llm=LLM(model="gemini/gemini-2.0-flash")
            )

            parser_agent = Agent(
                role="Data Extractor", 
                goal="Identify test names and numeric values from raw text.",
                backstory="Specializes in regex parsing and structure detection in documents.",
                tools=[parser_tool],
                allow_delegation=False,
                llm=LLM(model="gemini/gemini-2.0-flash")
            )

            analyzer_agent = Agent(
                role="Medical Analyst",
                goal="Analyze the parsed test results and flag abnormalities.",
                backstory="Doctor who evaluates lab data against clinical reference ranges.",
                tools=[analyzer_tool],
                allow_delegation=False,
                llm=LLM(model="gemini/gemini-2.0-flash")
            )

            summary_agent = Agent(
                role="Layman Summary Generator",
                goal="Convert clinical analysis into a friendly summary",
                backstory="You're a patient-friendly AI trained to explain lab reports simply.",
                tools=[summary_tool],
                allow_delegation=False,
                llm=LLM(model="gemini/gemini-2.0-flash")
            )

            # Add this after your existing agents
            # translation_agent = Agent(
            #     role="Medical Translator",
            #     goal="Translate medical analysis results to user's preferred language",
            #     backstory="Expert medical translator who makes health information accessible in multiple languages.",
            #     tools=[TranslationTool()],
            #     allow_delegation=False,
            #     llm=LLM(model="gemini/gemini-2.0-flash")
            # )

            # Define tasks
            task1 = Task(
                description="Extract text from the medical report located at {file_path}",
                agent=ocr_agent,
                expected_output="Raw medical text extracted from the document"
            )

            task2 = Task(
                description="Parse test values from the extracted text. Use the raw text from the previous task.", 
                agent=parser_agent, 
                expected_output="JSON string containing parsed test names and values", 
                context=[task1]
            )

            task3 = Task(
                description="Analyze the parsed test results for abnormalities. Use the JSON data from the previous task.", 
                agent=analyzer_agent, 
                expected_output="JSON string showing analysis of normal vs abnormal test results", 
                context=[task2]
            )

            task4 = Task(
                description=f"Convert the analysis JSON into a simple, friendly summary. {'Translate the summary to ' + target_lang_code if translate_option and target_lang_code != 'en' else 'Provide summary in English'}. Avoid medical jargon. Output bullet points.",
                agent=summary_agent,
                expected_output="Layman's summary of the medical analysis" + (f" in {target_lang_code}" if translate_option and target_lang_code != "en" else ""),
                context=[task3]
            )

            # Add translation task after task4
            # task5 = Task(
            #     description="Translate the medical summary to {target_language}. Keep medical terms accurate and maintain clarity.",
            #     agent=translation_agent,
            #     expected_output="Translated medical summary in the target language",
            #     context=[task4]  # Uses summary from task4
            # )

                    # Create single crew
            crew = Crew(
                agents=[ocr_agent, parser_agent, analyzer_agent, summary_agent],
                tasks=[task1, task2, task3, task4],
                verbose=True
            )
            
            result = crew.kickoff(inputs={"file_path": temp_path, "target_language": target_lang_code if translate_option else "en"})
            # Update your crew setup in the analyze button section
            # if translate_option and target_lang_code != "en":
            #     crew = Crew(
            #         agents=[ocr_agent, parser_agent, analyzer_agent, summary_agent, translation_agent],
            #         tasks=[task1, task2, task3, task4, task5],
            #         verbose=True
            #     )
            #     result = crew.kickoff(inputs={"file_path": temp_path, "target_language": target_lang_code})
            # else:
            #     crew = Crew(
            #         agents=[ocr_agent, parser_agent, analyzer_agent, summary_agent],
            #         tasks=[task1, task2, task3, task4],
            #         verbose=True
            #     )
            #     result = crew.kickoff(inputs={"file_path": temp_path})

            
        st.success("✅ Analysis Complete!")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 📊 Analysis Results")
            st.markdown(result.raw)
        
        # Get the analysis data from task3 output
        analysis_data = None
        if hasattr(result, 'tasks_output') and len(result.tasks_output) >= 3:
            analysis_data = result.tasks_output[2].raw  # Task 3 output
            print(f"DEBUG - Analysis data type: {type(analysis_data)}")
            print(f"DEBUG - Analysis data content: {analysis_data}")
            
            # Validate that analysis_data is not None and contains valid JSON
            if analysis_data and analysis_data.strip():
                try:
                    # Test if it's valid JSON
                    test_data = json.loads(analysis_data)
                    print(f"DEBUG - Parsed JSON successfully: {len(test_data)} items")
                except json.JSONDecodeError as e:
                    print(f"DEBUG - JSON parsing failed: {e}")
                    analysis_data = None
            else:
                print("DEBUG - Analysis data is None or empty")
                analysis_data = None

        with col2:
            st.markdown("## 📈 Visualizations")
            
            if analysis_data:
                # Create and display charts
                bar_chart = create_test_results_chart(analysis_data)
                if bar_chart:
                    st.plotly_chart(bar_chart, use_container_width=True)
                
                pie_chart = create_status_pie_chart(analysis_data)
                if pie_chart:
                    st.plotly_chart(pie_chart, use_container_width=True)
                
                # Display raw data table
                try:
                    data = json.loads(analysis_data)
                    if "error" not in data:
                        df = pd.DataFrame.from_dict(data, orient='index')
                        st.markdown("### 📋 Test Results Table")
                        st.dataframe(df, use_container_width=True)
                except:
                    pass
        
        # Generate and display conclusion
        if analysis_data:
            conclusion = generate_conclusion(analysis_data, result.raw)
            st.markdown("---")
            st.markdown(conclusion)
        
        # Download options
        st.markdown("---")
        st.markdown("## 💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.download_button(
                label="📄 Download Summary",
                data=result.raw,
                file_name="medical_analysis_summary.txt",
                mime="text/plain"
            ):
                st.success("Summary downloaded!")
        
        with col2:
            if analysis_data:
                if st.download_button(
                    label="📊 Download Data (JSON)",
                    data=analysis_data,
                    file_name="medical_test_data.json",
                    mime="application/json"
                ):
                    st.success("Data downloaded!")
        
        with col3:
            if analysis_data:
                conclusion_text = generate_conclusion(analysis_data, result.raw)
                if st.download_button(
                    label="📋 Download Full Report",
                    data=f"{result.raw}\n\n{conclusion_text}",
                    file_name="complete_medical_report.md",
                    mime="text/markdown"
                ):
                    st.success("Full report downloaded!")

        os.remove(temp_path)
        


# Add sidebar with information
with st.sidebar:
    st.markdown("## ℹ️ About This Tool")
    st.markdown("""
    This AI-powered medical report analyzer uses multiple specialized agents to:
    
    1. **Extract** text from your medical reports
    2. **Parse** test names and values
    3. **Analyze** results against normal ranges
    4. **Summarize** findings in simple language
    5. **Visualize** results with interactive charts
    6. **Generate** comprehensive conclusions
    
    ### 🔒 Privacy & Security
    - Files are processed locally
    - No data is stored permanently
    - Results are for informational purposes only
    
    ### ⚠️ Important Notice
    This tool is for educational purposes only and should not replace professional medical advice. Always consult with your healthcare provider for medical decisions.
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Powered By")
    st.markdown("- CrewAI Multi-Agent Framework")
    st.markdown("- Gemini AI Models")
    st.markdown("- Streamlit Interface")
    st.markdown("- Plotly Visualizations")

# Footer
st.markdown("""
<hr style="border: 1px solid #ccc;">
<p style='text-align:center; color:gray; font-size:14px'>
Made with ❤️ by <b>Saswata</b> 
</p>
""", unsafe_allow_html=True)
