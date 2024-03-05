import os
import requests
import tempfile
import shutil
from flask import Flask, request, redirect, url_for, jsonify, render_template, send_from_directory
import openai
import feedparser
import urllib.parse
from PyPDF2 import PdfReader
import mimetypes

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = os.getenv("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")

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Adjust the model as necessary
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
    )
    if response.choices and response.choices[0].message:
        return response.choices[0].message['content'].strip()
    else:
        return "No response generated."


def perform_search(search_query):
    base_url = 'http://export.arxiv.org/api/query?'
    query = f'search_query=all:{urllib.parse.quote(search_query)}&start=0&max_results=10'
    response = requests.get(base_url + query)
    feed = feedparser.parse(response.text)
    return [{'title': entry.title, 'id': entry.id.split('/abs/')[-1]} for entry in feed.entries]

def download_and_analyze_pdf(arxiv_id, question):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url)
    response.raise_for_status()
    pdf_content = response.content

    # Save the PDF content to a temporary file
    with open("temp.pdf", "wb") as f:
        f.write(pdf_content)

    # Read the PDF using PdfReader
    with open("temp.pdf", "rb") as f:
        pdf = PdfReader(f)
        text = ""
        # Use the pages attribute of PdfReader
        for page_num in range(min(10, len(pdf.pages))):  # Limiting to first 10 pages for brevity
            text += pdf.pages[page_num].extract_text()

    # Generate an answer using OpenAI
    prompt = f"Question: {question}\n\nText: {text}\n\nAnswer:"
    answer = generate_response(prompt)
    return answer

def main():
    query = input("Enter your search query: ")
    results = perform_search(query)
    for idx, article in enumerate(results, start=1):
        print(f"{idx}. Title: {article['title']}, arXiv ID: {article['id']}")

    arxiv_id_input = input("\nEnter the arXiv ID of the article to analyze: ")
    selected_article = next((article for article in results if article['id'] == arxiv_id_input), None)

    if selected_article:
        question = input("\nEnter your question about the selected article: ")
        answer = download_and_analyze_pdf(selected_article['id'], question)
        print("\nAnswer:", answer)
    else:
        print("Invalid arXiv ID. Please make sure you've entered the ID exactly as it appears in the list.")

if __name__ == '__main__':
    main()