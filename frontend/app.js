const API_BASE = "/api/v1";
const WINDOW_SIZE = 173;

const state = {
  signal: [],
  health: null,
};

const elements = {
  healthStatus: document.getElementById("healthStatus"),
  healthDetail: document.getElementById("healthDetail"),
  modelStatus: document.getElementById("modelStatus"),
  modelDetail: document.getElementById("modelDetail"),
  alertCountHero: document.getElementById("alertCountHero"),
  alertDetailHero: document.getElementById("alertDetailHero"),
  serviceStatusBadge: document.getElementById("serviceStatusBadge"),
  deviceValue: document.getElementById("deviceValue"),
  versionValue: document.getElementById("versionValue"),
  versionDetail: document.getElementById("versionDetail"),
  cnnVersions: document.getElementById("cnnVersions"),
  refreshTelemetry: document.getElementById("refreshTelemetry"),
  refreshAlerts: document.getElementById("refreshAlerts"),
  generateSignal: document.getElementById("generateSignal"),
  runPrediction: document.getElementById("runPrediction"),
  copyWindowJson: document.getElementById("copyWindowJson"),
  presetSelect: document.getElementById("presetSelect"),
  amplitudeInput: document.getElementById("amplitudeInput"),
  frequencyInput: document.getElementById("frequencyInput"),
  spikeInput: document.getElementById("spikeInput"),
  predictionWaveform: document.getElementById("predictionWaveform"),
  predictionStatus: document.getElementById("predictionStatus"),
  predictionConfidence: document.getElementById("predictionConfidence"),
  predictionThreshold: document.getElementById("predictionThreshold"),
  alertsList: document.getElementById("alertsList"),
  uploadForm: document.getElementById("uploadForm"),
  edfFile: document.getElementById("edfFile"),
  selectedFileName: document.getElementById("selectedFileName"),
  uploadResult: document.getElementById("uploadResult"),
  modelShapeBanner: document.getElementById("modelShapeBanner"),
  uploadWarning: document.getElementById("uploadWarning"),
  cursorHalo: document.getElementById("cursorHalo"),
  alzForm: document.getElementById("alzForm"),
  alzFile: document.getElementById("alzFile"),
  alzFileName: document.getElementById("alzFileName"),
  alzResult: document.getElementById("alzResult"),
  neuroForm: document.getElementById("neuroForm"),
  neuroFile: document.getElementById("neuroFile"),
  neuroFileName: document.getElementById("neuroFileName"),
  neuroResult: document.getElementById("neuroResult"),
  parkinsonPayload: document.getElementById("parkinsonPayload"),
  predictParkinson: document.getElementById("predictParkinson"),
  parkinsonResult: document.getElementById("parkinsonResult"),
  refreshObjectives: document.getElementById("refreshObjectives"),
  objectivesSummary: document.getElementById("objectivesSummary"),
  resultsList: document.getElementById("resultsList"),
};

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json();
}

function setServicePill(label, tone) {
  elements.serviceStatusBadge.textContent = label;
  elements.serviceStatusBadge.className = `inline-chip status-pill ${tone}`;
}

function applyTilt(card, event) {
  const rect = card.getBoundingClientRect();
  const px = (event.clientX - rect.left) / rect.width;
  const py = (event.clientY - rect.top) / rect.height;
  const rotateY = (px - 0.5) * 14;
  const rotateX = (0.5 - py) * 12;
  card.style.transform =
    `perspective(900px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
}

function resetTilt(card) {
  card.style.transform = "";
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function formatConfidence(value) {
  if (typeof value !== "number") return "--";
  return `${(value * 100).toFixed(2)}%`;
}

function formatTimestamp(unixSeconds) {
  if (!unixSeconds) return "Unknown time";
  return new Date(unixSeconds * 1000).toLocaleString();
}

function drawSignal(signal) {
  if (!signal.length) return;
  const width = 600;
  const height = 180;
  const maxMagnitude = Math.max(...signal.map((value) => Math.abs(value)), 1);
  const points = signal
    .map((value, index) => {
      const x = (index / (signal.length - 1)) * width;
      const y = height / 2 - (value / maxMagnitude) * (height * 0.38);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  elements.predictionWaveform.innerHTML = `<polyline points="${points}"></polyline>`;
}

function generateSignal() {
  const preset = elements.presetSelect.value;
  const amplitude = Number(elements.amplitudeInput.value);
  const frequency = Number(elements.frequencyInput.value);
  const spikeIntensity = Number(elements.spikeInput.value);
  const signal = [];

  for (let i = 0; i < WINDOW_SIZE; i += 1) {
    const t = i / WINDOW_SIZE;
    const base =
      amplitude * Math.sin(2 * Math.PI * frequency * t) +
      0.22 * amplitude * Math.sin(2 * Math.PI * (frequency * 0.5) * t + 0.6) +
      randomBetween(-0.12, 0.12);

    let value = base;

    if (preset === "spike") {
      if (i % 24 === 0 || i % 37 === 0) {
        value += spikeIntensity * randomBetween(1.4, 2.2);
      }
    } else if (preset === "ictal") {
      value += 0.7 * amplitude * Math.sin(2 * Math.PI * (frequency * 1.8) * t);
      if (i % 18 === 0) {
        value += spikeIntensity * randomBetween(0.8, 1.6);
      }
    } else {
      value += 0.08 * Math.sin(2 * Math.PI * 2 * t);
    }

    signal.push(Number(clamp(value, -4, 4).toFixed(5)));
  }

  state.signal = signal;
  drawSignal(signal);
}

async function refreshTelemetry() {
  elements.healthStatus.textContent = "Checking...";
  elements.modelStatus.textContent = "Checking...";
  elements.healthDetail.textContent = "Connecting to service";
  elements.modelDetail.textContent = "Loading active model metadata";
  setServicePill("Syncing", "muted");

  try {
    const [health, models, alerts] = await Promise.all([
      fetchJson("/health"),
      fetchJson("/models"),
      fetchJson("/alerts"),
    ]);

    const cnnVersions = Array.isArray(models.cnn) && models.cnn.length
      ? models.cnn.join(", ")
      : "None";
    state.health = health;

    elements.healthStatus.textContent = health.status === "ok" ? "Operational" : health.status;
    elements.healthDetail.textContent = `Device: ${health.device}`;
    elements.modelStatus.textContent = health.model_loaded ? "Loaded" : "Missing";
    elements.modelDetail.textContent = health.model_loaded
      ? `Version ${health.version} available for inference`
      : "No active checkpoint detected";
    elements.deviceValue.textContent = `Device ${health.device ?? "unknown"}`;
    elements.versionValue.textContent = `Model ${health.version ?? "unknown"}`;
    elements.versionDetail.textContent = health.version ?? "Unknown";
    elements.cnnVersions.textContent = cnnVersions;
    elements.modelShapeBanner.textContent = health.model_loaded
      ? `Active model expects ${health.expected_channels} channel(s) with window size ${health.expected_window_size}${health.dataset ? ` • dataset: ${health.dataset}` : ""}.`
      : "No active model loaded.";
    elements.alertCountHero.textContent = String(alerts.total ?? 0);
    elements.alertDetailHero.textContent = alerts.total
      ? "Recent alert history available"
      : "No alerts recorded";

    if (health.model_loaded && health.expected_channels === 1) {
      elements.uploadWarning.className = "upload-warning warn";
      elements.uploadWarning.textContent =
        "The active checkpoint is a single-channel Bonn model. Multi-channel EDF recordings will be rejected unless a compatible multi-channel checkpoint is deployed.";
    } else {
      elements.uploadWarning.className = "upload-warning";
      elements.uploadWarning.textContent =
        "EDF uploads are validated against the active model configuration before inference begins.";
    }

    if (health.status === "ok" && health.model_loaded) {
      setServicePill("Live", "ok");
    } else if (health.status === "ok") {
      setServicePill("Partial", "warn");
    } else {
      setServicePill("Offline", "error");
    }

    renderAlerts(alerts.alerts || []);
  } catch (error) {
    elements.healthStatus.textContent = "Offline";
    elements.healthDetail.textContent = "Service unavailable";
    elements.modelStatus.textContent = "Unavailable";
    elements.modelDetail.textContent = "Start the application on port 8000";
    elements.deviceValue.textContent = "Device unavailable";
    elements.versionValue.textContent = "Model unavailable";
    elements.versionDetail.textContent = "Unavailable";
    elements.cnnVersions.textContent = "Unavailable";
    elements.modelShapeBanner.textContent = "Model requirements unavailable.";
    elements.uploadWarning.className = "upload-warning";
    elements.uploadWarning.textContent = "Service telemetry unavailable.";
    elements.alertCountHero.textContent = "--";
    elements.alertDetailHero.textContent = "Alert telemetry unavailable";
    setServicePill("Offline", "error");
    renderAlerts([]);
    console.error(error);
  }
}

function renderAlerts(alerts) {
  if (!alerts.length) {
    elements.alertsList.innerHTML = '<div class="empty-state">No alerts yet.</div>';
    return;
  }

  elements.alertsList.innerHTML = alerts
    .slice()
    .reverse()
    .map((alert) => `
      <article class="alert-card">
        <header>
          <strong>${alert.status}</strong>
          <span class="status-pill ${alert.status === "ALERT" ? "warn" : "ok"}">${formatConfidence(alert.confidence)}</span>
        </header>
        <p>Threshold: ${alert.threshold}</p>
        <p>Timestamp: ${formatTimestamp(alert.timestamp)}</p>
        <p>Window Start: ${alert.window_start_sample ?? "n/a"}</p>
      </article>
    `)
    .join("");
}

async function runPrediction() {
  if (!state.signal.length) {
    generateSignal();
  }

  elements.predictionStatus.textContent = "Running...";
  elements.predictionConfidence.textContent = "--";
  elements.predictionThreshold.textContent = "--";

  try {
    const payload = {
      window: [state.signal],
      sfreq: 173.6,
    };

    const response = await fetchJson("/predict_window", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    elements.predictionStatus.textContent = response.status;
    elements.predictionConfidence.textContent = formatConfidence(response.confidence);
    elements.predictionThreshold.textContent = response.threshold.toFixed(2);
    await refreshTelemetry();
  } catch (error) {
    elements.predictionStatus.textContent = "Request failed";
    elements.predictionConfidence.textContent = "Unavailable";
    elements.predictionThreshold.textContent = "--";
    console.error(error);
  }
}

async function copyWindowJson() {
  if (!state.signal.length) {
    generateSignal();
  }

  const payload = JSON.stringify({
    window: [state.signal],
    sfreq: 173.6,
  }, null, 2);

  try {
    await navigator.clipboard.writeText(payload);
    elements.predictionStatus.textContent = "Payload copied";
  } catch (error) {
    console.error(error);
    elements.predictionStatus.textContent = "Copy unavailable";
  }
}

function renderUploadResult(message, tone = "") {
  elements.uploadResult.className = `upload-result ${tone}`.trim();
  elements.uploadResult.innerHTML = message;
}

function renderPanelResult(target, message, tone = "") {
  target.className = `upload-result ${tone}`.trim();
  target.innerHTML = message;
}

async function handleUpload(event) {
  event.preventDefault();
  const file = elements.edfFile.files?.[0];
  if (!file) {
    renderUploadResult("<p class=\"helper-copy\">Select an EDF file before submitting.</p>", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  renderUploadResult("<p class=\"helper-copy\">Uploading recording and starting analysis...</p>");

  try {
    const response = await fetch(`${API_BASE}/upload_edf`, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Upload failed");
    }

    renderUploadResult(`
      <p><strong>${payload.filename}</strong></p>
      <p>Channels: ${payload.n_channels}</p>
      <p>Samples: ${payload.n_samples}</p>
      <p>Duration: ${payload.duration_sec}s</p>
      <p>Annotated seizures: ${payload.n_seizures_annotated}</p>
      <p>${payload.message}</p>
    `, "success");

    await refreshTelemetry();
  } catch (error) {
    renderUploadResult(`<p class="helper-copy">${error.message}</p>`, "error");
    console.error(error);
  }
}

async function handleImageDisorderPredict(formEvent, input, target, endpoint) {
  formEvent.preventDefault();
  const file = input.files?.[0];
  if (!file) {
    renderPanelResult(target, "<p class=\"helper-copy\">Please select an image file.</p>", "error");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  renderPanelResult(target, "<p class=\"helper-copy\">Running prediction...</p>");
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Prediction failed");
    renderPanelResult(
      target,
      `<p><strong>${payload.prediction}</strong></p><p>Confidence: ${formatConfidence(payload.confidence)}</p>`,
      "success",
    );
    await refreshObjectives();
  } catch (error) {
    renderPanelResult(target, `<p class="helper-copy">${error.message}</p>`, "error");
  }
}

async function predictParkinsons() {
  let payload = {};
  try {
    payload = JSON.parse(elements.parkinsonPayload.value || "{}");
  } catch (error) {
    renderPanelResult(elements.parkinsonResult, "<p class=\"helper-copy\">Invalid JSON payload.</p>", "error");
    return;
  }

  renderPanelResult(elements.parkinsonResult, "<p class=\"helper-copy\">Running prediction...</p>");
  try {
    const response = await fetchJson("/predict/parkinsons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderPanelResult(
      elements.parkinsonResult,
      `<p><strong>${response.prediction}</strong></p><p>Confidence: ${formatConfidence(response.confidence)}</p><p>Target value: ${response.target_value}</p>`,
      "success",
    );
    await refreshObjectives();
  } catch (error) {
    renderPanelResult(elements.parkinsonResult, `<p class="helper-copy">${error.message}</p>`, "error");
  }
}

async function refreshObjectives() {
  try {
    const [modelInfo, datasetInfo, results] = await Promise.all([
      fetchJson("/model_info"),
      fetchJson("/dataset_info"),
      fetchJson("/results"),
    ]);

    elements.objectivesSummary.innerHTML = `
      <div class="status-row"><span>Alzheimer classes</span><strong>${(modelInfo.alzheimers.classes || []).join(", ") || "n/a"}</strong></div>
      <div class="status-row"><span>Parkinson features</span><strong>${modelInfo.parkinsons.feature_count ?? 0}</strong></div>
      <div class="status-row"><span>Neuro classes</span><strong>${(modelInfo.neuro.classes || []).join(", ") || "n/a"}</strong></div>
      <div class="status-row"><span>Total predictions</span><strong>${results.total ?? 0}</strong></div>
      <div class="status-row"><span>Alzheimer samples</span><strong>${datasetInfo.alzheimers.samples ?? 0}</strong></div>
      <div class="status-row"><span>Neuro samples</span><strong>${datasetInfo.neuro.samples ?? 0}</strong></div>
    `;

    const rows = (results.results || []).slice().reverse();
    elements.resultsList.innerHTML = rows.length
      ? rows
          .map(
            (row) => `
              <article class="alert-card">
                <header><strong>${row.disorder}</strong><span class="status-pill ok">${formatConfidence(row.confidence)}</span></header>
                <p>Prediction: ${row.prediction}</p>
                <p>Timestamp: ${formatTimestamp(row.timestamp)}</p>
              </article>
            `,
          )
          .join("")
      : '<div class="empty-state">No disorder predictions yet.</div>';
  } catch (error) {
    elements.objectivesSummary.innerHTML = '<div class="status-row"><span>Dashboard</span><strong>Unavailable</strong></div>';
    elements.resultsList.innerHTML = '<div class="empty-state">Unable to load results.</div>';
  }
}

function bindInteractions() {
  document.querySelectorAll(".tilt-card").forEach((card) => {
    card.addEventListener("mousemove", (event) => applyTilt(card, event));
    card.addEventListener("mouseleave", () => resetTilt(card));
  });

  document.addEventListener("pointermove", (event) => {
    elements.cursorHalo.style.left = `${event.clientX}px`;
    elements.cursorHalo.style.top = `${event.clientY}px`;
  });

  elements.refreshTelemetry.addEventListener("click", refreshTelemetry);
  elements.refreshAlerts.addEventListener("click", refreshTelemetry);
  elements.generateSignal.addEventListener("click", generateSignal);
  elements.runPrediction.addEventListener("click", runPrediction);
  elements.copyWindowJson.addEventListener("click", copyWindowJson);
  elements.uploadForm.addEventListener("submit", handleUpload);
  elements.edfFile.addEventListener("change", () => {
    const file = elements.edfFile.files?.[0];
    elements.selectedFileName.textContent = file ? file.name : "No file selected";
  });
  [elements.presetSelect, elements.amplitudeInput, elements.frequencyInput, elements.spikeInput]
    .forEach((element) => element.addEventListener("input", generateSignal));

  elements.alzForm?.addEventListener("submit", (event) =>
    handleImageDisorderPredict(event, elements.alzFile, elements.alzResult, "/predict/alzheimers"),
  );
  elements.neuroForm?.addEventListener("submit", (event) =>
    handleImageDisorderPredict(event, elements.neuroFile, elements.neuroResult, "/predict/neuro"),
  );
  elements.predictParkinson?.addEventListener("click", predictParkinsons);
  elements.refreshObjectives?.addEventListener("click", refreshObjectives);
  elements.alzFile?.addEventListener("change", () => {
    const file = elements.alzFile.files?.[0];
    elements.alzFileName.textContent = file ? file.name : "No file selected";
  });
  elements.neuroFile?.addEventListener("change", () => {
    const file = elements.neuroFile.files?.[0];
    elements.neuroFileName.textContent = file ? file.name : "No file selected";
  });
}

bindInteractions();
generateSignal();
refreshTelemetry();
refreshObjectives();
setInterval(refreshTelemetry, 15000);
