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
st.markdown("Upload a business document (PDF, Excel) to extract key metrics and generate an automated analysis at incredible speed.")

# --- File Type Selection ---
file_type = st.selectbox(
    "Select document type:",
    ["PDF Document", "Excel Spreadsheet"],
    index=0
)

# --- File Uploader ---
if file_type == "PDF Document":
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload annual reports, financial statements, business documents, etc."
    )
    file_icon = "📄"
    description = "Perfect for analyzing annual reports, financial statements, business presentations, and text-based documents."
    
else:  # Excel Spreadsheet
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help="Upload spreadsheets with business data, financial records, sales data, etc."
    )
    file_icon = "📊"
    description = "Ideal for analyzing sales data, financial records, inventory reports, customer data, and other structured business data."

# --- Display file type info ---
st.info(f"{file_icon} **{file_type}**: {description}")

if uploaded_file is not None:
    # Save the uploaded file to the 'documents' directory
    file_path = os.path.join("documents", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    # Show file details
    file_size = len(uploaded_file.getbuffer())
    st.markdown(f"**File size:** {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    # --- Processing and Displaying Results ---
    if st.button("🚀 Analyze Document", type="primary"):
        with st.spinner("Analyzing document... This will be quick! 🏎️"):
            # Call the main processing function from the backend
            result = processor.process_document(file_path)

        if result["success"]:
            st.balloons()
            st.header("✅ Analysis Complete!")

            data = result["data"]

            # Display Executive Summary
            st.subheader("📋 Executive Summary")
            st.info(data.get("executive_summary", "Not available."))

            # Display Key Metrics in a table
            st.subheader("📊 Key Insights & Metrics")
            metrics_df = pd.DataFrame(data.get("key_metrics", []))
            
            if not metrics_df.empty:
                st.dataframe(
                    metrics_df, 
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "metric": st.column_config.TextColumn("📈 Metric", width="medium"),
                        "value": st.column_config.TextColumn("💰 Value", width="small"), 
                        "commentary": st.column_config.TextColumn("💡 Commentary", width="large")
                    }
                )
                
                # Show metrics count
                st.caption(f"Found {len(metrics_df)} key metrics")
            else:
                st.warning("No metrics were extracted from the document.")

            # --- Download Buttons for Reports ---
            st.subheader("📥 Download Your Reports")
            col1, col2 = st.columns(2)

            with col1:
                try:
                    with open(result["excel_path"], "rb") as file:
                        st.download_button(
                            label="📊 Download Excel Report",
                            data=file,
                            file_name=os.path.basename(result["excel_path"]),
                            mime="application/vnd.ms-excel",
                            use_container_width=True
                        )
                except FileNotFoundError:
                    st.error("Excel report file not found.")

            with col2:
                try:
                    with open(result["word_path"], "rb") as file:
                        st.download_button(
                            label="📝 Download Word Report",
                            data=file,
                            file_name=os.path.basename(result["word_path"]),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                except FileNotFoundError:
                    st.error("Word report file not found.")
        else:
            st.error(f"❌ Analysis failed: {result['error']}")

# --- Sidebar Information ---
st.sidebar.header("🎯 About this App")
st.sidebar.write("""
This app demonstrates a RAG pipeline for document intelligence, powered by Groq's blazing-fast LPU inference engine.

**Supported formats:**
- 📄 **PDF**: Business reports, financial statements, presentations
- 📊 **Excel**: Spreadsheets, data tables, financial records

**Backend:** Python, Groq API, Pandas  
**Frontend:** Streamlit  

*Built for the Neostats Hackathon*
""")

st.sidebar.header("⚙️ How it Works")

if file_type == "PDF Document":
    st.sidebar.markdown("""
    **PDF Processing:**
    1. 📄 **Upload PDF**: You upload a document
    2. 🔍 **Text Extraction**: System reads and extracts all text
    3. 🤖 **AI Analysis**: Text sent to **Groq** (Llama 3.3) for analysis
    4. 📊 **Smart Chunking**: Large docs split intelligently
    5. 📈 **Report Generation**: Creates Excel and Word reports
    """)
else:
    st.sidebar.markdown("""
    **Excel Processing:**
    1. 📊 **Upload Excel**: You upload a spreadsheet
    2. 🔍 **Data Analysis**: System analyzes all sheets and columns
    3. 📈 **Statistical Insights**: Calculates key statistics
    4. 🤖 **AI Analysis**: Data summary sent to **Groq** for insights
    5. 📋 **Report Generation**: Creates comprehensive reports
    """)

st.sidebar.header("💡 Tips for Best Results")
st.sidebar.markdown("""
**For PDFs:**
- Use clear, text-based documents
- Financial reports work exceptionally well
- Scanned documents may have limited accuracy

**For Excel:**
- Include headers in your data
- Use meaningful column names
- Multiple sheets are fully supported
- Financial/business data yields best insights
""")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        ⚡️ Powered by <strong>Groq</strong> | Built with ❤️ for Neostats Hackathon
    </div>
    """, 
    unsafe_allow_html=True
)