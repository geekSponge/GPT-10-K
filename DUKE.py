import streamlit as st
import openai
import json
import fitz  # PyMuPDF
from openai import AzureOpenAI
import os
import base64
import chromadb
from chromadb.utils import embedding_functions

# Read configuration data from the JSON file
with open(r'C:\users\h\downloads\config.json') as config_file:
    config_data = json.load(config_file)

# Select a specific configuration
selected_config = config_data['openAIConfigs'][0]

print(f"Using configuration: {selected_config['configName']}")

# Define the model name to be used
model_name = selected_config['model']

# Initialize the Azure OpenAI client
azure_client = AzureOpenAI(
    api_key=selected_config['apiKey'],
    api_version=selected_config['apiVersion'],
    azure_endpoint=selected_config['azureEndpoint']
)

# Function to load and parse JSON file
def load_json_file(file):
    data = json.load(file)
    return data['analyzeResult']['content']

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    content = ""
    for page in pdf_document:
        content += page.get_text()
    return content

# Function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to create markdown description for images
def image_to_markdown(base64_image):
    prompt = "Try the best to answer question based on the info we provided. Describe the following picture as precisely as you can. It should contain all the information such that someone can recreate the image from the text explanation. Convert tables to markdown tables. Describe charts as best you can. Don't interpret what you see, only describe, nothing else. DO NOT return in a codeblock. Just return the raw text in markdown format."
    response = azure_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "user", "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
        ],
        max_tokens=4096
    )
    return response.choices[0].message.content

# Function to query GPT with context
def query_gpt_with_context(context, question):
    behaviour = "You are a financial data analyst specialized in quantitative analysis of annual reports and financial statements Read and analyze chapter by chapter the content of the json file. The following is an annual report you need to analyze stored in a json file/ pdf file only. Don't use any knowledge from external resources."
    response = azure_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": behaviour},
            {"role": "user", "content": f"CONTEXT: {context}\n\nQUESTION: {question}"}
        ],
        temperature=0.5,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )
    return response.choices[0].message.content

# Streamlit app
def main():
    st.title("Annual Report Analyzer")
    
    # File uploader for JSON or PDF file
    uploaded_file = st.file_uploader("Upload a JSON or PDF file", type=["json", "pdf"])
    
    if uploaded_file:
        file_type = uploaded_file.type
        if file_type == "application/json":
            content = load_json_file(uploaded_file)
        elif file_type == "application/pdf":
            content = extract_text_from_pdf(uploaded_file)
        
        st.write("Annual report loaded successfully.")
        
        # Generate a vector database for the document
        chroma_client = chromadb.PersistentClient("/content/.chromadb")
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        collection = chroma_client.get_or_create_collection(name="documents", embedding_function=default_ef)
        collection.upsert(documents=[content], ids=["document1"])

        # Ask the user for the specific information they need
        question = st.text_input("What info do you need from this annual report?")
        
        if st.button("Get Answer"):
            with st.spinner("Analyzing the report..."):
                # Query the vector database and generate a context-based response
                results = collection.query(query_texts=[question], n_results=1)
                context = results['documents'][0]
                answer = query_gpt_with_context(context, question)
                st.success("Answer:")
                st.write(answer)
                st.write("Source Data:")
                st.code(content, language='markdown')

        # Process images if it is a PDF file
        if file_type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            image_folder = "image_folder"
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
                
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images()
                for image_index, img in enumerate(image_list, start=1):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    output_path = f"{image_folder}/{uploaded_file.name}_page_{page_index+1}-image_{image_index}.png"
                    pix.save(output_path)
                    base64_image = encode_image_to_base64(output_path)
                    markdown_content = image_to_markdown(base64_image)
                    st.markdown(f"## Page {page_index+1}, Image {image_index}")
                    st.markdown(markdown_content)

if __name__ == "__main__":
    main()