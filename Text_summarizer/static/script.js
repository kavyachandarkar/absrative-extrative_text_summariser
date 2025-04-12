// MAIN summarize button click
document.getElementById('summarize-btn').addEventListener('click', summarize);

// Scroll to section
function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: "smooth" });
}

// Update summary length label
function updateLengthLabel(value) {
  document.getElementById("lengthLabel").innerText = value;
}

// Theme Toggle
function toggleTheme() {
  document.body.classList.toggle("dark-mode");
}

// Show Toast Notification
function showToast(message) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => container.removeChild(toast), 3000);
}

// Copy summary to clipboard
function copySummary() {
  const text = document.getElementById("summary").textContent;
  navigator.clipboard.writeText(text).then(() => {
    showToast("📋 Summary copied to clipboard!");
  });
}

// Download summary as .txt file
function downloadSummary() {
  const text = document.getElementById("summary").textContent;
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "summary.txt";
  a.click();
}

// Display evaluation scores
function displayScores(data) {
  const { rouge_scores, bleu_score, precision_recall } = data;
  const scoreBox = document.getElementById("scoreDisplay");

  scoreBox.innerHTML = `
    <h4>Evaluation Scores</h4>
    <p>ROUGE-1: ${rouge_scores["ROUGE-1"].toFixed(3)}</p>
    <p>ROUGE-2: ${rouge_scores["ROUGE-2"].toFixed(3)}</p>
    <p>ROUGE-L: ${rouge_scores["ROUGE-L"].toFixed(3)}</p>
    <p>BLEU Score: ${bleu_score.toFixed(2)}</p>
    <p>Precision: ${(precision_recall.Precision * 100).toFixed(2)}%</p>
    <p>Recall: ${(precision_recall.Recall * 100).toFixed(2)}%</p>
  `;
}

// Add summary to history
function addToHistory(summary) {
  const list = document.getElementById("historyList");
  const li = document.createElement("li");
  li.textContent = summary.slice(0, 80) + "...";
  list.appendChild(li);
}

// Core summarize function
async function summarize() {
  const textInput = document.getElementById("inputText").value.trim();
  const fileInput = document.getElementById("fileInput").files[0];
  const summaryBox = document.getElementById("summary");
  const loader = document.getElementById("loader");
  const statusMessage = document.getElementById("statusMessage");
  const summaryType = document.getElementById("summaryType").value;
  const maxLength = document.getElementById("maxLength").value || 130;
  const minLength = document.getElementById("minLength").value || 30;
  const progressBar = document.getElementById("progressBar");

  summaryBox.textContent = "";
  statusMessage.textContent = "";
  progressBar.style.display = "block";
  loader.style.display = "block";

  try {
    let response;
    if (fileInput) {
      const formData = new FormData();
      formData.append("file", fileInput);
      formData.append("type", summaryType);
      formData.append("max_length", maxLength);
      formData.append("min_length", minLength);

      response = await fetch("/summarize", {
        method: "POST",
        body: formData
      });
    } else if (textInput) {
      response = await fetch("/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: textInput,
          type: summaryType,
          max_length: parseInt(maxLength),
          min_length: parseInt(minLength)
        })
      });
    } else {
      statusMessage.textContent = "⚠️ Please provide text or upload a PDF.";
      loader.style.display = "none";
      progressBar.style.display = "none";
      return;
    }

    const data = await response.json();
    loader.style.display = "none";
    progressBar.style.display = "none";

    if (data.error) {
      statusMessage.textContent = `⚠️ ${data.error}`;
    } else {
      summaryBox.textContent = data.summary;
      addToHistory(data.summary);
      displayScores(data);
      showToast("✅ Summary generated successfully!");
    }
  } catch (err) {
    loader.style.display = "none";
    progressBar.style.display = "none";
    statusMessage.textContent = "❌ An error occurred.";
    console.error(err);
  }
}
// Translation Button Click Handler
document.getElementById("translate-btn").addEventListener("click", async () => {
    const summaryText = document.getElementById("summary").textContent.trim();
  
    if (!summaryText) {
      alert("Please generate a summary first before translating.");
      return;
    }
  
    try {
      const response = await fetch("/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: summaryText })
      });
  
      const data = await response.json();
  
      if (response.ok) {
        document.getElementById("translated-text").textContent = data.translated_text; // ✅ use 'translated_text' as per backend
        showToast("🌐 Translation completed!");
      } else {
        document.getElementById("translated-text").textContent = "Translation failed: " + data.error;
      }
    } catch (error) {
      document.getElementById("translated-text").textContent = "Translation error: " + error.message;
    }
  });
  
  

// Modal close handler
function closeModal() {
  document.getElementById("modal").style.display = "none";
}

// Close modal on outside click
window.onclick = function (event) {
  const modal = document.getElementById("modal");
  if (event.target == modal) {
    modal.style.display = "none";
  }
};
