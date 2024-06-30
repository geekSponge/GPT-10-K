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
selected_config = config_data['openAIConfigs'][2]

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

# Function to chunk text
def chunk_text(text, max_tokens=120000):
    # Estimate the maximum number of words per chunk
    max_words = int(max_tokens / 1.5)
    
    # Split the text into words
    words = text.split()
    
    # Calculate the number of chunks needed
    num_chunks = len(words) // max_words + (1 if len(words) % max_words != 0 else 0)
    
    # Create the chunks
    chunks = [" ".join(words[i * max_words: (i + 1) * max_words]) for i in range(num_chunks)]
    
    return chunks

# Function to query GPT with context
def query_gpt_with_context(context, question):
    behaviour = "You are a financial data analyst specialized in quantitative analysis of annual reports and financial statements. Read and analyze chapter by chapter the content of the json file. The following is an annual report you need to analyze stored in a json file/ pdf file only. Don't use any knowledge from external resources. Please remember that when you give us the answer. Please make sure the text is neatly formatted in appearance. Also, after the complete answer, please always provide the appendix (where you find the answer, show me the exactly originally contexts where your answer is based on). About the part please give me a new paragraph and highlight and start with SOURCE DATA. The format of answers on the whole should make sense and look reasonable and decent. Last but not least, the source data need to be complete, in detail, solid and make the answer convincing as well as stay true to the original."
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

def query_gpt_with_combined_answer(context, question):
    behaviour = r"""You are a financial data analyst specialized in quantitative analysis of annual reports and financial statements. 
    You are giving input from multiple chunks. Please give answer based on that. 
    Don't use any knowledge from external resources. 
    Please remember that when you give us the answer, make sure the text is neatly formatted in appearance. 
    Also, after the complete answer, please always provide the appendix (where you find the answer, show me the exactly originally contexts where your answer is based on). About the part please give me a new paragraph and highlight and start with SOURCE DATA. 
    The format of answers on the whole should make sense and look reasonable and decent. 
    Frmat of sentences should be clear and conform to uniform standard(ASCII characters). Pay special attention to the combinition of digital number and units. Make them look normal like their neighbors
    Last but not least, the source data need to be complete, in detail, solid and make the answer convincing as well as stay true to the original."""
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
                # Chunk the content
                chunks = chunk_text(content)
                # st.success("Chunk number:")
                # st.write(len(chunks))
                # Query the GPT model with each chunk and combine responses
                combined_answer = ""
                for chunk in chunks:
                    chunk_answer = query_gpt_with_context(chunk, question)
                    combined_answer += chunk_answer + "\n"
                
                if len(chunks) > 1:
                    final_answer = query_gpt_with_combined_answer(combined_answer, question)
                else:
                    final_answer = combined_answer
                st.success("Answer:")
                st.write(final_answer)

if __name__ == "__main__":
    main()
