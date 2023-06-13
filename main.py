import requests
import os
import csv
import shutil
import tempfile
import re
from flask import Flask, render_template, request, redirect, url_for, jsonify
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.schema import HumanMessage
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.document_loaders import PyPDFLoader
import random
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import feedparser
import urllib.parse
import PyPDF2
import requests

import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import send_from_directory
import mimetypes


app = Flask(__name__)

openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

llm = OpenAI(temperature=0.75)
chat = ChatOpenAI(temperature=0.75)


@app.route('/process_message', methods=['POST'])
def process_message():
    # Process the user's message and generate a response
    message = request.json['message']
    response = generate_response(message)

    return jsonify({'message': response})


def get_questions(prompt):
    response = chat([HumanMessage(content=prompt)])
    questions = [response.content.strip()]
    return questions

def get_answers(question, abstract):
    messages = [HumanMessage(content=question), HumanMessage(content=f"abstract: {abstract}")]
    response = chat(messages)
    answer = response.content.strip()
    return answer

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf = PyPDF2.PdfFileReader(f)
        text = ""
        for page_num in range(pdf.numPages):
            page = pdf.getPage(page_num)
            text += page.extractText()
        return text


def process_words_csv(file_path):
    words = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            word = row[1]
            words.append(word)
    return words


def get_falling_words():
    file_path = 'templates/static/words.csv'
    words = process_words_csv(file_path)
    return words


def split_text(text, max_length):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]



def get_arxiv_results(search_query, start=0, max_results=20):
    base_url = 'http://export.arxiv.org/api/query?'
    query = f'search_query=all:{urllib.parse.quote(search_query)}&start={start}&max_results={max_results}'
    response = urllib.request.urlopen(base_url + query).read()
    feed = feedparser.parse(response)
    return feed


def parse_arxiv_results(feed):
    entries = feed.entries
    results = []
    for entry in entries:
        result = {
            "arxiv_id": entry.id.split('/abs/')[-1],
            "title": entry.title,
            "authors": entry.author,
            "abstract": entry.summary
        }
        results.append(result)
    return results


def download_pdf(selected_paper_id):
    pdf_url = f"https://arxiv.org/pdf/{selected_paper_id}.pdf"
    response = requests.get(pdf_url)
    response.raise_for_status()

    # Create the 'temp' directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)

    temp_file = f"temp/{selected_paper_id}.pdf"
    with open(temp_file, "wb") as f:
        f.write(response.content)
    return temp_file


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' in request.files:
            # Get the uploaded PDF file
            pdf_file = request.files['pdf_file']

            # Save the uploaded PDF file to a temporary location
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, pdf_file.filename)
            pdf_file.save(temp_file_path)

            # Clean up the temporary files
            shutil.rmtree(temp_dir)

            return redirect(url_for('pdf_chat', pdf_path=temp_file_path))
        else:
            search_query = request.form.get('search_query')
            if search_query:
                return redirect(url_for('result', search_query=search_query))
    return render_template('index.html')


def perform_search(search_query):
    try:
        feed = get_arxiv_results(search_query)
        results = parse_arxiv_results(feed)
        return results
    except Exception as e:
        print(f"An error occurred during the search: {str(e)}")
        return []



def qa(file_or_arxiv_id, query, chain_type, k):
    if os.path.isfile(file_or_arxiv_id):
        # File path provided
        file_path = file_or_arxiv_id
        arxiv_id = os.path.splitext(os.path.basename(file_path))[0]
        if not arxiv_id.startswith("arXiv"):
            arxiv_id = os.path.splitext(os.path.basename(file_path))[0].split(".")[0]
    else:
        # arXiv ID provided
        arxiv_id = file_or_arxiv_id
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url)
        response.raise_for_status()

        # Save the PDF file
        file_path = f"{arxiv_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(response.content)

    if not os.path.isfile(file_path):
        raise ValueError(f"File path {file_path} is not a valid file")

    # Extract text from the PDF
    text = extract_text_from_pdf(file_path)

    # Split the document into chunks
    texts = [text]

    # Select embeddings to use
    embeddings = OpenAIEmbeddings()

    # Create the vectorstore to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # Expose the index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Create a chain to answer questions
    qa_model = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type=chain_type, retriever=retriever,
                                          return_source_documents=True)

    # Perform question answering
    result = qa_model({"query": query})
    answers = result["result"]

    # Format the answers
    formatted_answers = []
    for answer in answers:
        text = answer["answer"]["value"]
        confidence = answer["answer"]["confidence"]
        source_document = answer["source_document"]
        formatted_answer = {
            "text": text,
            "confidence": confidence,
            "source_document": source_document,
        }
        formatted_answers.append(formatted_answer)

    return formatted_answers


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        search_query = request.form.get('search_query')
        return redirect(url_for('result', search_query=search_query))
    else:
        search_query = request.args.get('search_query')
        # Perform search and retrieve the results based on the search_query
        results = perform_search(search_query)

        # Generate the figure
        fig = plt.figure()
        # ... add plots or other elements to the figure ...

        # Save the figure to an in-memory buffer
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.getvalue()).decode()

        return render_template('result.html', results=results, fig=img_data)


@app.route('/qa', methods=['POST'])
def perform_qa():
    arxiv_id = request.json['arxivId']
    question = request.json['question']
    chain_type = "map_rerank"
    k = 2
    answer = qa(arxiv_id, question, chain_type, k)
    return jsonify({'answer': answer})



@app.route('/genie', methods=['GET', 'POST'])
def genie():
    chat_log = []

    if request.method == 'POST':
        arxiv_id = request.form.get('arxiv_id')
        question = request.form['question']
        chain_type = "map_rerank"  # Set the desired chain_type
        k = 1  # Set the desired value for k

        if arxiv_id:
            # Perform question answering for the provided arXiv ID
            answers = qa(arxiv_id, question, chain_type, k)

            # Add the inquiry and answers to the chat log
            chat_log.append({"text": question, "is_question": True})
            for answer in answers:
                chat_log.append({"text": answer["text"], "is_question": False})
        else:
            # No arXiv ID provided, show an error message
            chat_log.append({"text": "Please provide a valid arXiv ID.", "is_question": False})

    return render_template('genie.html', chat_log=chat_log)


@app.route('/pdf_upload', methods=['GET', 'POST'])
def pdf_upload():
    if request.method == 'POST':
        # Get the uploaded PDF file
        pdf_file = request.files['pdf_file']

        # Save the uploaded PDF file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, pdf_file.filename)
        pdf_file.save(temp_file_path)

        # Load the document
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Select embeddings to use
        embeddings = OpenAIEmbeddings()

        # Create the vectorstore to use as the index
        db = Chroma.from_documents(texts, embeddings)

        # Expose the index in a retriever interface
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        # Create a chain to answer questions
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="map_rerank", retriever=retriever, return_source_documents=True)

        # Clean up the temporary files
        shutil.rmtree(temp_dir)

        return redirect(url_for('pdf_chat'))
    else:
        return render_template('pdf_upload.html')


@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        question = request.form.get('question')
        chain_type = "map_rerank"  # Set the desired chain_type
        k = 2  # Set the desired value for k

        # Perform QA using the `qa` function
        answer = qa(pdf_file, question, chain_type, k)

        return render_template('pdf_chat.html', answer=answer)

    return render_template('pdf_chat.html')


@app.route('/pdf_chat/<path:filename>', methods=['GET'])
def pdf_chat_static(filename):
    mimetype, _ = mimetypes.guess_type(filename)
    return send_from_directory('static', filename, mimetype=mimetype)





def generate_response(user_message):
    # Generate the response based on the user message
    # Replace this with your own logic for generating responses
    return 'This is the response to your message: ' + user_message





if __name__ == '__main__':
    app.run(debug=True, port=5001)
