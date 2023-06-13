// pdf_chat.js

const chatboxContent = document.getElementById('chatbox-content');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');

function addMessage(message, isUserMessage = false) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');

    if (isUserMessage) {
        messageElement.classList.add('user-message');
    }

    messageElement.innerText = message;
    chatboxContent.appendChild(messageElement);
}

function sendMessage(event) {
    event.preventDefault();
    const userMessage = userInput.value.trim();

    if (userMessage !== '') {
        // Add user message to the chatbox
        addMessage(userMessage, true);
        userInput.value = '';

        // Send the user message to the server for processing
        sendUserMessage(userMessage);
    }
}

function receiveMessage(message) {
    // Add the received message to the chatbox
    addMessage(message);
}

function sendUserMessage(message) {
    // Send the user message to the server using AJAX or Fetch API
    // Replace the URL with the appropriate endpoint for processing the user message
    fetch('/process_message', {
        method: 'POST',
        body: JSON.stringify({ message: message }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        // Receive the response from the server and add it to the chatbox
        receiveMessage(data.message);
    })
    .catch(error => {
        console.log('An error occurred:', error);
    });
}

chatForm.addEventListener('submit', sendMessage);
