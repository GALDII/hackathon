import os
import json
from groq import Groq
from dotenv import load_dotenv
import pypdf
import pandas as pd
from docx import Document

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client
try:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    print(f"Reading PDF from: {pdf_path}")
    with open(pdf_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    print("Successfully extracted text from PDF.")
    return text

def get_insights_from_llm(text_content):
    """
    Sends the extracted text to a current Groq LLM to get structured insights.
    """
    if not client:
        raise ConnectionError("Groq client not initialized. Check your API key.")

    # This prompt instructs the LLM to act as an analyst and return a specific JSON structure.
    system_prompt = """
    You are a highly skilled financial analyst AI. Your task is to analyze the provided text from a business document and extract key performance indicators (KPIs), metrics, and a summary.

    Please provide the output in a structured JSON format ONLY. Do not include any introductory text, explanations, or markdown formatting outside of the JSON structure itself.

    The JSON structure must be as follows:
    {
      "executive_summary": "A concise, 2-3 sentence summary of the overall business performance based on the document.",
      "key_metrics": [
        {
          "metric": "Metric Name (e.g., Total Revenue, Net Profit, YoY Growth)",
          "value": "The value of the metric (e.g., '$150 Million', '12.5%')",
          "commentary": "A brief, one-sentence comment on the significance of this metric."
        }
      ]
    }

    Extract at least 5-7 of the most important metrics you can find in the text.
    """

    print("Sending text to Groq LLM for analysis...")
    try:
        response = client.chat.completions.create(
            # FIXED: Using correct current model name
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the document text:\n\n{text_content}"}
            ]
        )
        json_response = response.choices[0].message.content
        print("Received structured JSON data from Groq LLM.")
        return json.loads(json_response)
    except Exception as e:
        print(f"An error occurred while communicating with the Groq API: {e}")
        return None

def create_excel_report(data, output_path):
    """Creates an Excel report from the extracted data."""
    df = pd.DataFrame(data['key_metrics'])
    df = df[['metric', 'value', 'commentary']]
    df.to_excel(output_path, index=False)
    print(f"Excel report saved to {output_path}")

def create_word_report(data, output_path):
    """Creates a Word document report from the extracted data."""
    doc = Document()
    doc.add_heading('Business Performance Analysis Report', level=1)
    
    doc.add_heading('Executive Summary', level=2)
    summary = data.get('executive_summary', 'No summary available.')
    doc.add_paragraph(summary)
    
    doc.add_heading('Key Metrics', level=2)
    metrics = data.get('key_metrics', [])
    
    if metrics:
        table = doc.add_table(rows=1, cols=3)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Metric'
        hdr_cells[1].text = 'Value'
        hdr_cells[2].text = 'Commentary'
        
        for item in metrics:
            row_cells = table.add_row().cells
            row_cells[0].text = item.get('metric', '')
            row_cells[1].text = item.get('value', '')
            row_cells[2].text = item.get('commentary', '')
    else:
        doc.add_paragraph("No key metrics were extracted.")
        
    doc.save(output_path)
    print(f"Word report saved to {output_path}")

def process_document(file_path):
    """Main orchestration function."""
    try:
        document_text = extract_text_from_pdf(file_path)
        if not document_text:
            raise ValueError("Could not extract text from the PDF.")

        insights_data = get_insights_from_llm(document_text)
        if not insights_data:
            raise ValueError("Failed to get insights from the language model.")
        
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        excel_path = os.path.join("outputs", f"{base_filename}_report.xlsx")
        word_path = os.path.join("outputs", f"{base_filename}_report.docx")

        create_excel_report(insights_data, excel_path)
        create_word_report(insights_data, word_path)

        return {
            "success": True,
            "data": insights_data,
            "excel_path": excel_path,
            "word_path": word_path
        }
    except Exception as e:
        print(f"An error occurred during document processing: {e}")
        return {"success": False, "error": str(e)}