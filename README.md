# GPT-10-K
# DUKE_multi.py - Annual Report Analyzer
## Overview
DUKE_multi.py is a Streamlit-based web application designed to analyze multiple annual reports in JSON or PDF format using OpenAI's GPT-4 model. The application allows users to upload multiple 10-K files and provides detailed answers to specific questions based on the content of the reports.

## Features
- **Multi-file Upload**: Users can upload multiple JSON or PDF files simultaneously for analysis.
- **Text Extraction**: The app extracts text from the uploaded files. For JSON files, it parses and retrieves the relevant content. For PDF files, it utilizes PyMuPDF to extract text from each page.
- **Content Chunking**: The extracted content is chunked into manageable parts to ensure efficient processing by the GPT-4 model.
- **AI-Powered Analysis**: The application leverages Azure's OpenAI service to analyze the content of the reports and provide answers to user queries.
- **Detailed Responses**: The GPT-4 model provides detailed answers along with source data from the original documents, ensuring transparency and traceability of the analysis.
- **Persistent Vector Database**: The application uses ChromaDB to store and manage embeddings of the combined document content for efficient querying and analysis.

## Usage
1. **Upload Files**: Users can upload one or more annual reports in JSON or PDF format.
2. **Enter Query**: Users input specific questions they need answers to, based on the content of the uploaded reports.
3. **Analyze**: Upon clicking the "Get Answer" button, the application processes the reports, queries the GPT-4 model, and displays the answers along with the source data.

## Requirements
- `streamlit`
- `openai`
- `PyMuPDF`
- `azure-openai`
- `chromadb`
- `json`
- `os`
- `base64`

## How to Run
1. Ensure all dependencies are installed as per the `requirements.txt`.
2. Place the logo image (`logo.png`) in the same directory as `DUKE_multi.py`.
3. Run the application using Streamlit:
   ```sh
   streamlit run DUKE_multi.py
