document.addEventListener("DOMContentLoaded", function() {
  const form = document.querySelector("form");
  const log = document.getElementById("log");

  form.addEventListener("submit", function(event) {
    event.preventDefault();
    const formData = new FormData(form);
    const arxivId = formData.get("arxiv_id");
    const question = formData.get("question");

    if (arxivId && question) {
      // Clear the log
      log.innerHTML = "";

      // Display the user's question in the log
      addLogEntry("User", question);

      // Make an AJAX request to the server to get the response
      makeRequest(arxivId, question);
    }
  });

  function makeRequest(arxivId, question) {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/genie");
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    xhr.onload = function() {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);
        const answer = response.answer;
        // Display the Genie's answer in the log
        addLogEntry("Genie", answer);
      }
    };
    xhr.send(`arxiv_id=${encodeURIComponent(arxivId)}&question=${encodeURIComponent(question)}`);
  }

  function addLogEntry(user, content) {
    const entry = document.createElement("div");
    entry.classList.add("log-entry");
    entry.innerHTML = `<p class="question">${user}: ${content}</p>`;
    log.appendChild(entry);
  }
});
