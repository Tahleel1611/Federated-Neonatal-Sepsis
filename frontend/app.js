const apiUrlInput = document.getElementById("apiUrl");
const payloadInput = document.getElementById("payload");
const resultOutput = document.getElementById("result");
const loadExampleButton = document.getElementById("loadExample");
const predictButton = document.getElementById("predict");

function setResult(value) {
  resultOutput.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function loadExamplePayload() {
  const baseUrl = apiUrlInput.value.replace(/\/$/, "");
  const response = await fetch(`${baseUrl}/example-payload`);
  if (!response.ok) {
    throw new Error(`Example payload request failed: ${response.status}`);
  }
  const payload = await response.json();
  payloadInput.value = JSON.stringify(payload, null, 2);
  setResult("Example payload loaded.");
}

async function runPrediction() {
  const baseUrl = apiUrlInput.value.replace(/\/$/, "");
  const payload = JSON.parse(payloadInput.value);
  const response = await fetch(`${baseUrl}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `Prediction request failed: ${response.status}`);
  }
  setResult(body);
}

loadExampleButton.addEventListener("click", async () => {
  try {
    await loadExamplePayload();
  } catch (error) {
    setResult(String(error.message || error));
  }
});

predictButton.addEventListener("click", async () => {
  try {
    setResult("Running prediction...");
    await runPrediction();
  } catch (error) {
    setResult(String(error.message || error));
  }
});
