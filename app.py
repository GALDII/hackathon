import streamlit as st
import os
import pandas as pd
import processor

# Create directories if they don't exist
os.makedirs("documents", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Neostats Doc Intelligence",
    page_icon="⚡️",
    layout="wide"
)

st.title("⚡️ Groq-Powered Document Intelligence")
st.markdown("Upload a business document (e.g., annual sales report PDF) to extract key metrics and generate an automated analysis at incredible speed.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf"
)

if uploaded_file is not None:
    # Save the uploaded file to the 'documents' directory
    file_path = os.path.join("documents", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # --- Processing and Displaying Results ---
    if st.button("🚀 Analyze Document"):
        with st.spinner("Analyzing document... This will be quick! 🏎️"):
            # Call the main processing function from the backend
            result = processor.process_document(file_path)

        if result["success"]:
            st.balloons()
            st.header("Analysis Complete!")

            data = result["data"]

            # Display Executive Summary
            st.subheader("Executive Summary")
            st.info(data.get("executive_summary", "Not available."))

            # Display Key Metrics in a table
            st.subheader("Key Metrics Extracted")
            metrics_df = pd.DataFrame(data.get("key_metrics", []))
            st.dataframe(metrics_df, use_container_width=True)

            # --- Download Buttons for Reports ---
            st.subheader("Download Your Reports")
            col1, col2 = st.columns(2)

            with col1:
                try:
                    with open(result["excel_path"], "rb") as file:
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=file,
                            file_name=os.path.basename(result["excel_path"]),
                            mime="application/vnd.ms-excel"
                        )
                except FileNotFoundError:
                    st.error("Excel report file not found.")

            with col2:
                try:
                    with open(result["word_path"], "rb") as file:
                        st.download_button(
                            label="📥 Download Word Report",
                            data=file,
                            file_name=os.path.basename(result["word_path"]),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                except FileNotFoundError:
                    st.error("Word report file not found.")
        else:
            st.error(f"Analysis failed: {result['error']}")

# --- Sidebar Information ---
st.sidebar.header("About this App")
st.sidebar.write("""
This app demonstrates a RAG pipeline for document intelligence, powered by Groq's blazing-fast LPU inference engine.
- **Backend:** Python, Groq API, Pandas
- **Frontend:** Streamlit
It's designed as a template for the Neostats Hackathon.
""")
st.sidebar.header("How it Works")
st.sidebar.markdown("""
1.  **Upload PDF**: You upload a document.
2.  **Text Extraction**: The system reads and extracts all text from the PDF.
3.  **AI Analysis**: The text is sent to **Groq** to be analyzed by Llama 3.1, which pulls out key metrics in a structured JSON format.
4.  **Report Generation**: The data is used to create Excel and Word reports.
""")