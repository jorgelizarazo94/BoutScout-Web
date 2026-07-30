/* BoutScout browser inference, multi-file review, and visual editing. */

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

const COLORS = { Nocturnal: "#333E48", Off: "#535AA6", On: "#E28342", Unlabeled: "#e2e8f0" };
const LABEL_MAP = { 0: "Nocturnal", 1: "Off", 2: "On" };
const SENSOR_ORDER = { egg: 0, nest: 1 };

let modelSession = null;
let modelLoadError = null;
let analyses = new Map();
let nextSourceOrder = 0;
let currentAnalysisKey = null;
let currentDay = null;
let currentEvents = [];
let isEditMode = false;
let selectedEventId = null;
let selectedRange = null;
let creationMode = false;
let pendingBoundaryRange = null;
let visibleRange = [0, 1440];

function logDebug(message) {
    const logBox = document.getElementById("debugLog");
    if (!logBox) return;
    logBox.innerHTML += `> ${message}<br>`;
    logBox.scrollTop = logBox.scrollHeight;
}

function showNotification(text, isError = false) {
    const notification = document.getElementById("notification");
    const icon = notification.querySelector("i");
    document.getElementById("notificationText").innerText = text;
    icon.className = isError ? "ph ph-warning-circle text-red-400 text-2xl" : "ph ph-check-circle text-green-400 text-2xl";
    notification.classList.remove("translate-y-20", "opacity-0");
    window.setTimeout(() => notification.classList.add("translate-y-20", "opacity-0"), 4000);
}

const modelReadyPromise = (async function loadModel() {
    try {
        const response = await fetch("model_final_bilstm.onnx");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const buffer = await response.arrayBuffer();
        ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
        modelSession = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
        logDebug("Modelo ONNX listo.");
        return modelSession;
    } catch (error) {
        modelLoadError = error;
        logDebug(`No se pudo cargar el modelo: ${error.message}`);
        return null;
    }
})();

function normalizeHeader(value) {
    return String(value || "").replace(/^\uFEFF/, "").trim().toLowerCase().replace(/[\s_]+/g, "-");
}

function findColumn(headers, acceptedNames) {
    const normalized = headers.map(normalizeHeader);
    return normalized.findIndex(header => acceptedNames.includes(header));
}

function parseCsvFile(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: false,
            skipEmptyLines: true,
            complete: results => resolve(results.data),
            error: reject
        });
    });
}

function parseCsvUrl(url) {
    return new Promise((resolve, reject) => {
        Papa.parse(url, {
            download: true,
            header: false,
            skipEmptyLines: true,
            complete: results => resolve(results.data),
            error: reject
        });
    });
}

function getCurrentAnalysis() {
    return analyses.get(currentAnalysisKey) || null;
}

function getCurrentDayData() {
    const analysis = getCurrentAnalysis();
    return analysis && currentDay ? analysis.days[currentDay] : null;
}

function removeExistingSource(sourceFile) {
    for (const [key, analysis] of analyses.entries()) {
        if (analysis.sourceFile === sourceFile) analyses.delete(key);
    }
}

async function processDataset(data, sourceFile, sourceOrder) {
    if (!Array.isArray(data) || data.length < 2) throw new Error("El CSV no contiene datos.");

    const headers = data[0];
    const dateIndex = findColumn(headers, ["date-time", "datetime", "date/time"]);
    const eggIndex = findColumn(headers, ["egg"]);
    const nestIndex = findColumn(headers, ["nest"]);
    const ambientIndex = findColumn(headers, ["ambient"]);

    if (dateIndex < 0) throw new Error("Falta la columna Date-time.");
    if (ambientIndex < 0) throw new Error("Falta la columna Ambient.");
    if (eggIndex < 0 && nestIndex < 0) throw new Error("Falta una columna egg o Nest.");

    const sensors = [];
    if (eggIndex >= 0) sensors.push({ type: "egg", index: eggIndex });
    if (nestIndex >= 0) sensors.push({ type: "nest", index: nestIndex });

    for (const sensor of sensors) {
        const groupedByDate = {};

        for (let index = 1; index < data.length; index += 1) {
            const row = data[index];
            if (!row) continue;

            const rawDate = row[dateIndex];
            const sensorTemperature = Number.parseFloat(row[sensor.index]);
            const ambientTemperature = Number.parseFloat(row[ambientIndex]);
            if (!rawDate || !Number.isFinite(sensorTemperature) || !Number.isFinite(ambientTemperature)) continue;

            const datetime = new Date(String(rawDate).trim());
            if (Number.isNaN(datetime.getTime())) {
                logDebug(`${sourceFile}: fecha inválida omitida (${rawDate}).`);
                continue;
            }

            const year = datetime.getFullYear();
            const month = String(datetime.getMonth() + 1).padStart(2, "0");
            const day = String(datetime.getDate()).padStart(2, "0");
            const date = `${year}-${month}-${day}`;
            const minuteOfDay = datetime.getHours() * 60 + datetime.getMinutes();
            const parsedRow = {
                source_file: sourceFile,
                sensor_type: sensor.type,
                date,
                datetime: datetime.toISOString(),
                minute_of_day: minuteOfDay,
                egg_temperature: sensorTemperature,
                sensor_temperature: sensorTemperature,
                ambient_temperature: ambientTemperature,
                hour_sin: Math.sin(2 * Math.PI * minuteOfDay / 1440),
                hour_cos: Math.cos(2 * Math.PI * minuteOfDay / 1440),
                minute_of_day_scaled: minuteOfDay / 1439,
                raw_label: "Unknown",
                final_label: "Unknown"
            };

            if (!groupedByDate[date]) groupedByDate[date] = [];
            groupedByDate[date].push(parsedRow);
        }

        const days = {};
        let skippedDays = 0;
        for (const date of Object.keys(groupedByDate).sort()) {
            const minuteMap = new Map();
            groupedByDate[date]
                .sort((a, b) => a.minute_of_day - b.minute_of_day)
                .forEach(row => {
                    if (!minuteMap.has(row.minute_of_day)) minuteMap.set(row.minute_of_day, row);
                });

            const cleanDay = Array.from(minuteMap.values()).sort((a, b) => a.minute_of_day - b.minute_of_day);
            const isComplete = cleanDay.length === 1440 && cleanDay[0].minute_of_day === 0 && cleanDay[1439].minute_of_day === 1439;
            if (!isComplete) {
                skippedDays += 1;
                logDebug(`${sourceFile} · ${sensor.type} · ${date}: ${cleanDay.length}/1440 minutos; día omitido.`);
                continue;
            }

            await runBiLSTM(cleanDay);
            days[date] = cleanDay;
        }

        const dates = Object.keys(days);
        if (dates.length > 0) {
            const key = `${sourceOrder}::${sourceFile}::${sensor.type}`;
            analyses.set(key, { key, sourceFile, sourceOrder, sensorType: sensor.type, days });
            logDebug(`${sourceFile} · ${sensor.type}: ${dates.length} días usados, ${skippedDays} omitidos.`);
        } else {
            logDebug(`${sourceFile} · ${sensor.type}: no hay días completos.`);
        }
    }
}

async function runBiLSTM(dayArray) {
    const floatArray = new Float32Array(1440 * 5);
    for (let index = 0; index < 1440; index += 1) {
        const row = dayArray[index];
        floatArray[index * 5] = row.sensor_temperature;
        floatArray[index * 5 + 1] = row.ambient_temperature;
        floatArray[index * 5 + 2] = row.hour_sin;
        floatArray[index * 5 + 3] = row.hour_cos;
        floatArray[index * 5 + 4] = row.minute_of_day_scaled;
    }

    const tensor = new ort.Tensor("float32", floatArray, [1, 1440, 5]);
    const feeds = { [modelSession.inputNames[0]]: tensor };
    const results = await modelSession.run(feeds);
    const outputData = results[modelSession.outputNames[0]].data;

    if (outputData.length < 1440 * 3) throw new Error("La salida del modelo no tiene el tamaño esperado.");
    for (let index = 0; index < 1440; index += 1) {
        let maxIndex = 0;
        let maxValue = outputData[index * 3];
        if (outputData[index * 3 + 1] > maxValue) {
            maxValue = outputData[index * 3 + 1];
            maxIndex = 1;
        }
        if (outputData[index * 3 + 2] > maxValue) maxIndex = 2;
        dayArray[index].raw_label = LABEL_MAP[maxIndex];
        dayArray[index].final_label = LABEL_MAP[maxIndex];
    }
}

async function processFiles(files) {
    if (!files.length) return;
    const loading = document.getElementById("loadingIndicator");
    loading.classList.remove("hidden");
    document.getElementById("initialMessage").classList.add("hidden");
    document.getElementById("csvNameDisplay").innerText = `${files.length} CSV`;

    try {
        await modelReadyPromise;
        if (!modelSession) throw modelLoadError || new Error("No se pudo iniciar el modelo.");
        const fileErrors = [];
        for (const file of files) {
            try {
                removeExistingSource(file.name);
                const rows = await parseCsvFile(file);
                await processDataset(rows, file.name, nextSourceOrder++);
            } catch (error) {
                fileErrors.push(`${file.name}: ${error.message}`);
                logDebug(`${file.name}: ${error.message}`);
            }
        }
        if (analyses.size === 0) throw new Error(fileErrors.join("\n") || "No se encontraron días completos de 1440 minutos.");
        refreshSourceSelector();
        if (fileErrors.length) showNotification(`${files.length - fileErrors.length} CSV procesados; ${fileErrors.length} con errores.`, true);
        else showNotification(`${files.length} CSV procesado${files.length === 1 ? "" : "s"}.`);
    } catch (error) {
        const message = modelLoadError ? "No se pudo iniciar el modelo de inferencia. Recarga la página e inténtalo de nuevo." : error.message;
        showNotification(message, true);
        alert(message);
        if (analyses.size === 0) document.getElementById("initialMessage").classList.remove("hidden");
    } finally {
        loading.classList.add("hidden");
    }
}

async function loadDemoData() {
    const loading = document.getElementById("loadingIndicator");
    loading.classList.remove("hidden");
    document.getElementById("initialMessage").classList.add("hidden");
    document.getElementById("csvNameDisplay").innerText = "demo_nest_processed.csv";
    try {
        await modelReadyPromise;
        if (!modelSession) throw modelLoadError || new Error("No se pudo iniciar el modelo.");
        removeExistingSource("demo_nest_processed.csv");
        const rows = await parseCsvUrl("data/demo_nest_processed.csv");
        await processDataset(rows, "demo_nest_processed.csv", nextSourceOrder++);
        refreshSourceSelector();
        showNotification("Datos demo procesados.");
    } catch (error) {
        const message = modelLoadError ? "No se pudo iniciar el modelo de inferencia." : error.message;
        showNotification(message, true);
        alert(message);
        document.getElementById("initialMessage").classList.remove("hidden");
    } finally {
        loading.classList.add("hidden");
    }
}

document.getElementById("csvFileInput").addEventListener("change", event => {
    processFiles(Array.from(event.target.files || []));
    event.target.value = "";
});

function orderedAnalyses() {
    return Array.from(analyses.values()).sort((a, b) =>
        a.sourceOrder - b.sourceOrder || SENSOR_ORDER[a.sensorType] - SENSOR_ORDER[b.sensorType]
    );
}

function refreshSourceSelector() {
    const sourceSelect = document.getElementById("sourceSelect");
    const previous = analyses.has(currentAnalysisKey) ? currentAnalysisKey : null;
    sourceSelect.innerHTML = "";
    orderedAnalyses().forEach(analysis => {
        const option = document.createElement("option");
        option.value = analysis.key;
        option.innerText = `${analysis.sourceFile} — ${analysis.sensorType === "egg" ? "Egg" : "Nest"}`;
        sourceSelect.appendChild(option);
    });

    document.getElementById("sourceSelectorContainer").classList.remove("hidden");
    document.getElementById("dateSelectorContainer").classList.remove("hidden");
    currentAnalysisKey = previous || orderedAnalyses()[0].key;
    sourceSelect.value = currentAnalysisKey;
    sourceSelect.onchange = event => selectAnalysis(event.target.value);
    populateDateSelector();
}

function selectAnalysis(key) {
    currentAnalysisKey = key;
    populateDateSelector();
}

function populateDateSelector() {
    const analysis = getCurrentAnalysis();
    if (!analysis) return;
    const dateSelect = document.getElementById("dateSelect");
    dateSelect.innerHTML = "";
    Object.keys(analysis.days).sort().forEach(date => {
        const option = document.createElement("option");
        option.value = date;
        option.innerText = date;
        dateSelect.appendChild(option);
    });
    dateSelect.onchange = event => loadDay(event.target.value);
    loadDay(Object.keys(analysis.days).sort()[0]);
}

function loadDay(date) {
    currentDay = date;
    visibleRange = [0, 1440];
    clearEditSelection();
    document.getElementById("workspace").classList.remove("hidden");
    const analysis = getCurrentAnalysis();
    document.getElementById("chartTitle").innerText = `${analysis.sourceFile} — ${analysis.sensorType === "egg" ? "Egg" : "Nest"} — ${date}`;
    recalculateEvents(analysis.days[date], false);
}

function recalculateEvents(dayData, preserveZoom = true) {
    currentEvents = [];
    let currentEvent = null;
    dayData.forEach(row => {
        if (!currentEvent || row.final_label !== currentEvent.label) {
            if (currentEvent) {
                currentEvent.mean_temp = (currentEvent.tempSum / currentEvent.count).toFixed(2);
                currentEvents.push(currentEvent);
            }
            currentEvent = { start_minute: row.minute_of_day, end_minute: row.minute_of_day, label: row.final_label, tempSum: row.sensor_temperature, count: 1 };
        } else {
            currentEvent.end_minute = row.minute_of_day;
            currentEvent.tempSum += row.sensor_temperature;
            currentEvent.count += 1;
        }
    });
    if (currentEvent) {
        currentEvent.mean_temp = (currentEvent.tempSum / currentEvent.count).toFixed(2);
        currentEvents.push(currentEvent);
    }
    currentEvents.forEach((event, index) => { event.id = index; });
    renderTable();
    renderChart(dayData, preserveZoom);
}

function chartShapes() {
    const shapes = currentEvents.map(event => ({
        type: "rect", xref: "x", yref: "paper",
        x0: event.start_minute, x1: event.end_minute + 1, y0: 0, y1: 1,
        fillcolor: COLORS[event.label] || COLORS.Unlabeled,
        opacity: event.id === selectedEventId ? 0.58 : (event.label === "Unlabeled" ? 0.15 : 0.25),
        line: { color: event.id === selectedEventId ? "#0f172a" : "transparent", width: event.id === selectedEventId ? 2 : 0 },
        layer: "below"
    }));

    const preview = pendingBoundaryRange || selectedRange;
    if (preview) {
        shapes.push({
            type: "rect", xref: "x", yref: "paper",
            x0: preview[0], x1: preview[1] + 1, y0: 0, y1: 1,
            fillcolor: pendingBoundaryRange ? "#3b82f6" : "#64748b",
            opacity: 0.28,
            line: { color: pendingBoundaryRange ? "#2563eb" : "#475569", width: 2, dash: "dash" },
            layer: "below"
        });
    }
    return shapes;
}

function renderChart(dayData, preserveZoom = true) {
    if (!dayData) return;
    const analysis = getCurrentAnalysis();
    const sensorName = analysis.sensorType === "egg" ? "Egg" : "Nest";
    const traces = [
        {
            x: dayData.map(row => row.minute_of_day), y: dayData.map(row => row.ambient_temperature),
            type: "scatter", mode: "lines", line: { color: "#94a3b8", width: 1.8 },
            name: "Ambient temp", hovertemplate: "Min: %{x}<br>Ambient: %{y:.1f}°C<extra></extra>"
        },
        {
            x: dayData.map(row => row.minute_of_day), y: dayData.map(row => row.sensor_temperature),
            type: "scatter", mode: "lines", line: { color: "#0f172a", width: 1.5 },
            name: `${sensorName} temp`, hovertemplate: `Min: %{x}<br>${sensorName}: %{y:.1f}°C<extra></extra>`
        }
    ];
    const range = preserveZoom ? visibleRange : [0, 1440];
    const layout = {
        margin: { t: 10, r: 10, b: 0, l: 40 }, hovermode: "x unified", showlegend: true,
        dragmode: isEditMode ? "select" : "zoom",
        xaxis: { title: "", range, rangeslider: { visible: true, thickness: 0.15, bgcolor: "#f1f5f9" } },
        yaxis: { fixedrange: true }, plot_bgcolor: "#ffffff", shapes: chartShapes(),
        selectionrevision: "boutscout-edit", uirevision: `${currentAnalysisKey}:${currentDay}`
    };
    const graph = document.getElementById("plotlyChart");
    Plotly.react(graph, traces, layout, { responsive: true, displayModeBar: false }).then(() => bindChartHandlers(graph));
}

function bindChartHandlers(graph) {
    if (graph.__boutScoutHandlersBound) return;
    graph.__boutScoutHandlersBound = true;
    graph.on("plotly_relayout", changes => {
        if (Array.isArray(changes["xaxis.range"])) visibleRange = changes["xaxis.range"].map(Number);
        else if (changes["xaxis.range[0]"] !== undefined && changes["xaxis.range[1]"] !== undefined) {
            visibleRange = [Number(changes["xaxis.range[0]"]), Number(changes["xaxis.range[1]"])];
        } else if (changes["xaxis.autorange"]) visibleRange = [0, 1440];
    });
    graph.on("plotly_selected", eventData => {
        if (!isEditMode || !eventData || !eventData.range || !eventData.range.x) return;
        selectedEventId = null;
        pendingBoundaryRange = null;
        selectedRange = [
            Math.max(0, Math.floor(eventData.range.x[0])),
            Math.min(1439, Math.ceil(eventData.range.x[1]))
        ];
        if (selectedRange[0] > selectedRange[1]) selectedRange.reverse();
        creationMode = true;
        refreshEditControls();
        renderChart(getCurrentDayData(), true);
    });
    graph.on("plotly_click", data => {
        if (!isEditMode || creationMode || !data.points || data.points.length === 0) return;
        const minute = data.points[0].x;
        const event = currentEvents.find(item => minute >= item.start_minute && minute <= item.end_minute);
        if (!event) return;
        selectedRange = null;
        pendingBoundaryRange = [event.start_minute, event.end_minute];
        selectedEventId = selectedEventId === event.id ? null : event.id;
        if (selectedEventId === null) pendingBoundaryRange = null;
        refreshEditControls();
        renderChart(getCurrentDayData(), true);
    });
}

function toggleEditMode() {
    isEditMode = !isEditMode;
    const button = document.getElementById("btnToggleEdit");
    const banner = document.getElementById("editModeBanner");
    if (isEditMode) {
        button.classList.add("editing-active");
        document.getElementById("btnEditOnText").innerText = t("btnEditOff");
        banner.classList.remove("hidden");
        banner.classList.add("flex");
    } else {
        button.classList.remove("editing-active");
        document.getElementById("btnEditOnText").innerText = t("btnEditOn");
        banner.classList.add("hidden");
        banner.classList.remove("flex");
        clearEditSelection();
    }
    renderChart(getCurrentDayData(), true);
}

function startNewLabel() {
    if (!isEditMode) toggleEditMode();
    creationMode = true;
    selectedEventId = null;
    selectedRange = null;
    pendingBoundaryRange = null;
    refreshEditControls();
    document.getElementById("selectedSegmentMsg").innerText = currentLang === "es" ? "Arrastra sobre la gráfica" : currentLang === "fr" ? "Faites glisser sur le graphique" : "Drag over the chart";
    Plotly.relayout("plotlyChart", { dragmode: "select" });
}

function clearEditSelection() {
    selectedEventId = null;
    selectedRange = null;
    creationMode = false;
    pendingBoundaryRange = null;
    refreshEditControls();
}

function cancelEditSelection() {
    clearEditSelection();
    renderChart(getCurrentDayData(), true);
}

function refreshEditControls() {
    const actions = document.getElementById("labelActions");
    const boundaryEditor = document.getElementById("boundaryEditor");
    const message = document.getElementById("selectedSegmentMsg");
    const newLabelButton = document.getElementById("btnNewLabel");
    const hasSelection = selectedRange !== null || selectedEventId !== null;
    actions.classList.toggle("hidden", !hasSelection);
    actions.classList.toggle("flex", hasSelection);
    boundaryEditor.classList.toggle("hidden", selectedEventId === null);
    newLabelButton.classList.toggle("ring-4", creationMode);
    newLabelButton.classList.toggle("ring-slate-300", creationMode);

    if (selectedEventId !== null) {
        const event = currentEvents.find(item => item.id === selectedEventId);
        if (event) {
            const range = pendingBoundaryRange || [event.start_minute, event.end_minute];
            document.getElementById("boundaryStart").value = range[0];
            document.getElementById("boundaryEnd").value = range[1];
            document.getElementById("boundaryValues").innerText = `${range[0]} – ${range[1]} min`;
            message.innerText = `${t("editBannerSelected")}: ${event.start_minute} – ${event.end_minute}`;
            return;
        }
    }
    if (selectedRange) {
        message.innerText = `${t("editBannerSelected")}: ${selectedRange[0]} – ${selectedRange[1]}`;
    } else if (!creationMode) {
        message.innerText = t("editBannerNone");
    }
}

function previewBoundaryChange() {
    if (selectedEventId === null) return;
    let start = Number(document.getElementById("boundaryStart").value);
    let end = Number(document.getElementById("boundaryEnd").value);
    if (start > end) [start, end] = [end, start];
    pendingBoundaryRange = [start, end];
    document.getElementById("boundaryValues").innerText = `${start} – ${end} min`;
    renderChart(getCurrentDayData(), true);
}

function applyBoundaryChange() {
    const event = currentEvents.find(item => item.id === selectedEventId);
    if (!event || !pendingBoundaryRange) return;
    const [newStart, newEnd] = pendingBoundaryRange;
    const previous = currentEvents[event.id - 1];
    const next = currentEvents[event.id + 1];
    const dayData = getCurrentDayData();

    dayData.forEach(row => {
        if (row.minute_of_day >= newStart && row.minute_of_day <= newEnd) row.final_label = event.label;
        else if (row.minute_of_day >= event.start_minute && row.minute_of_day < newStart) row.final_label = previous ? previous.label : "Unlabeled";
        else if (row.minute_of_day > newEnd && row.minute_of_day <= event.end_minute) row.final_label = next ? next.label : "Unlabeled";
    });
    clearEditSelection();
    recalculateEvents(dayData, true);
    showNotification("Límites actualizados.");
}

function applySelectedLabel(newLabel) {
    const dayData = getCurrentDayData();
    if (!dayData) return;
    let range = selectedRange;
    if (!range && selectedEventId !== null) {
        const event = currentEvents.find(item => item.id === selectedEventId);
        if (event) range = [event.start_minute, event.end_minute];
    }
    if (!range) return;
    dayData.forEach(row => {
        if (row.minute_of_day >= range[0] && row.minute_of_day <= range[1]) row.final_label = newLabel;
    });
    clearEditSelection();
    recalculateEvents(dayData, true);
    showNotification(`Aplicado: ${newLabel}`);
}

window.addEventListener("keydown", event => {
    if (!isEditMode || (!selectedRange && selectedEventId === null)) return;
    const labels = { "1": "On", "2": "Off", "3": "Nocturnal", Backspace: "Unlabeled", Delete: "Unlabeled" };
    if (!labels[event.key]) return;
    event.preventDefault();
    applySelectedLabel(labels[event.key]);
});

function renderTable() {
    const body = document.getElementById("eventsTableBody");
    body.innerHTML = "";
    currentEvents.forEach(event => {
        const row = document.createElement("tr");
        row.className = "hover:bg-slate-50 border-b border-slate-100 transition";
        const options = ["Nocturnal", "Off", "On", "Unlabeled"].map(label =>
            `<option value="${label}" ${event.label === label ? "selected" : ""}>${label}</option>`
        ).join("");
        const textColor = event.label === "Unlabeled" ? "text-slate-600" : "text-white";
        row.innerHTML = `
            <td class="px-5 py-3 font-mono text-xs text-slate-500">${event.start_minute}</td>
            <td class="px-5 py-3 font-mono text-xs text-slate-500">${event.end_minute}</td>
            <td class="px-5 py-3 font-bold text-slate-700">${event.end_minute - event.start_minute + 1} m</td>
            <td class="px-5 py-3 text-slate-600">${event.mean_temp} °C</td>
            <td class="px-5 py-3"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold ${textColor} shadow-sm" style="background-color:${COLORS[event.label]}">${event.label}</span></td>
            <td class="px-5 py-2.5"><select onchange="updateLabel(${event.id}, this.value)" class="w-full bg-white border border-blue-200 rounded-lg px-3 py-1.5 text-sm font-bold text-blue-900 outline-none cursor-pointer">${options}</select></td>`;
        body.appendChild(row);
    });
}

function updateLabel(eventId, newLabel) {
    selectedEventId = eventId;
    selectedRange = null;
    applySelectedLabel(newLabel);
}

function csvEscape(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    const stringValue = String(value);
    return /[,"\n]/.test(stringValue) ? `"${stringValue.replaceAll('"', '""')}"` : stringValue;
}

function downloadCSV(filename, rows, columns) {
    const lines = [columns.join(","), ...rows.map(row => columns.map(column => csvEscape(row[column])).join(","))];
    const blob = new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function buildMinuteRows() {
    const rows = [];
    orderedAnalyses().forEach(analysis => {
        Object.keys(analysis.days).sort().forEach(date => {
            analysis.days[date].forEach(row => rows.push({
                source_file: analysis.sourceFile, sensor_type: analysis.sensorType,
                date: row.date, datetime: row.datetime, minute_of_day: row.minute_of_day,
                egg_temperature: row.sensor_temperature, ambient_temperature: row.ambient_temperature,
                raw_label: row.raw_label, final_label: row.final_label
            }));
        });
    });
    return rows;
}

function mean(values) {
    const clean = values.filter(Number.isFinite);
    return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : NaN;
}

function minimum(values) {
    const clean = values.filter(Number.isFinite);
    return clean.length ? Math.min(...clean) : NaN;
}

function maximum(values) {
    const clean = values.filter(Number.isFinite);
    return clean.length ? Math.max(...clean) : NaN;
}

function standardDeviation(values) {
    const clean = values.filter(Number.isFinite);
    if (clean.length <= 1) return NaN;
    const average = mean(clean);
    return Math.sqrt(clean.reduce((sum, value) => sum + (value - average) ** 2, 0) / (clean.length - 1));
}

function finalizeEvent(event, eventNumber, counters, analysis) {
    const label = event.label || "Unknown";
    counters[label] = (counters[label] || 0) + 1;
    const sensorMean = mean(event.sensorValues);
    const ambientMean = mean(event.ambientValues);
    const sensorMin = minimum(event.sensorValues);
    const sensorMax = maximum(event.sensorValues);
    const ambientMin = minimum(event.ambientValues);
    const ambientMax = maximum(event.ambientValues);
    return {
        source_file: analysis.sourceFile, date: event.date, event_id: `${label}_${counters[label]}`,
        event_number: eventNumber, label, start_minute: event.startMinute, end_minute: event.endMinute,
        duration_min: event.endMinute - event.startMinute + 1, sensor_type: analysis.sensorType,
        mean_egg_temp: sensorMean, min_egg_temp: sensorMin, max_egg_temp: sensorMax,
        mean_ambient_temp: ambientMean, min_ambient_temp: ambientMin, max_ambient_temp: ambientMax,
        min_egg_temp_off: label === "Off" ? sensorMin : NaN,
        min_ambient_temp_off: label === "Off" ? ambientMin : NaN,
        mean_egg_temp_off: label === "Off" ? sensorMean : NaN,
        mean_ambient_temp_off: label === "Off" ? ambientMean : NaN,
        mean_egg_temp_on: label === "On" ? sensorMean : NaN,
        mean_ambient_temp_on: label === "On" ? ambientMean : NaN,
        mean_egg_temp_nocturnal: label === "Nocturnal" ? sensorMean : NaN,
        mean_ambient_temp_nocturnal: label === "Nocturnal" ? ambientMean : NaN
    };
}

function extractEventsFromDay(dayData, analysis) {
    const events = [];
    const counters = {};
    let current = null;
    let eventNumber = 1;
    dayData.forEach(row => {
        if (!current || row.final_label !== current.label) {
            if (current) events.push(finalizeEvent(current, eventNumber++, counters, analysis));
            current = { date: row.date, label: row.final_label || "Unknown", startMinute: row.minute_of_day, endMinute: row.minute_of_day, sensorValues: [row.sensor_temperature], ambientValues: [row.ambient_temperature] };
        } else {
            current.endMinute = row.minute_of_day;
            current.sensorValues.push(row.sensor_temperature);
            current.ambientValues.push(row.ambient_temperature);
        }
    });
    if (current) events.push(finalizeEvent(current, eventNumber, counters, analysis));
    return events;
}

function buildEventRows() {
    const events = [];
    orderedAnalyses().forEach(analysis => {
        Object.keys(analysis.days).sort().forEach(date => events.push(...extractEventsFromDay(analysis.days[date], analysis)));
    });
    return events;
}

function buildDailySummaryRows(events, minuteRows) {
    const eventGroups = {};
    const minuteGroups = {};
    events.forEach(event => {
        const key = `${event.source_file}::${event.sensor_type}::${event.date}`;
        (eventGroups[key] ||= []).push(event);
    });
    minuteRows.forEach(row => {
        const key = `${row.source_file}::${row.sensor_type}::${row.date}`;
        (minuteGroups[key] ||= []).push(row);
    });

    return Object.keys(eventGroups).map(key => {
        const group = eventGroups[key].sort((a, b) => a.start_minute - b.start_minute);
        const first = group[0];
        const on = group.filter(event => event.label === "On");
        const off = group.filter(event => event.label === "Off");
        const nocturnal = group.filter(event => event.label === "Nocturnal");
        let intervalStart = NaN;
        let intervalEnd = NaN;
        let intervalDuration = NaN;
        let totalOnInterval = NaN;
        let attentiveness = NaN;
        if (off.length && nocturnal.length) {
            intervalStart = off[0].start_minute;
            intervalEnd = nocturnal[nocturnal.length - 1].start_minute - 1;
            intervalDuration = intervalEnd - intervalStart + 1;
            if (intervalDuration > 0) {
                totalOnInterval = on.filter(event => event.start_minute >= intervalStart && event.end_minute <= intervalEnd).reduce((sum, event) => sum + event.duration_min, 0);
                attentiveness = totalOnInterval / intervalDuration;
            }
        }
        const minutes = minuteGroups[key] || [];
        const sensorValues = minutes.map(row => Number(row.egg_temperature));
        const ambientValues = minutes.map(row => Number(row.ambient_temperature));
        return {
            source_file: first.source_file, date: first.date, sensor_type: first.sensor_type,
            nest_attentiveness_day: attentiveness, duracion_total_on: totalOnInterval,
            duracion_intervalo_day: intervalDuration, inicio_intervalo: intervalStart, fin_intervalo: intervalEnd,
            total_on_full_day: on.reduce((sum, event) => sum + event.duration_min, 0),
            total_off_full_day: off.reduce((sum, event) => sum + event.duration_min, 0),
            total_nocturnal_full_day: nocturnal.reduce((sum, event) => sum + event.duration_min, 0),
            avg_duracion_on: mean(on.map(event => event.duration_min)), avg_duracion_off: mean(off.map(event => event.duration_min)),
            avg_duracion_nocturnal: mean(nocturnal.map(event => event.duration_min)),
            min_duracion_on: minimum(on.map(event => event.duration_min)), max_duracion_on: maximum(on.map(event => event.duration_min)),
            min_duracion_off: minimum(off.map(event => event.duration_min)), max_duracion_off: maximum(off.map(event => event.duration_min)),
            n_eventos_on: on.length, n_eventos_off: off.length, n_eventos_nocturnal: nocturnal.length,
            avg_temp_ambient_diaria: mean(ambientValues), std_temp_ambient_diaria: standardDeviation(ambientValues),
            avg_temp_sensor_diaria: mean(sensorValues), min_temp_sensor_diaria: minimum(sensorValues), max_temp_sensor_diaria: maximum(sensorValues),
            mean_min_egg_temp_off_bouts: mean(off.map(event => event.min_egg_temp_off)),
            lowest_egg_temp_recorded_during_off_bout: minimum(off.map(event => event.min_egg_temp_off)),
            mean_min_ambient_temp_off_bouts: mean(off.map(event => event.min_ambient_temp_off)),
            lowest_ambient_temp_recorded_during_off_bout: minimum(off.map(event => event.min_ambient_temp_off))
        };
    });
}

function exportAllResults() {
    const minuteRows = buildMinuteRows();
    const eventRows = buildEventRows();
    const dailyRows = buildDailySummaryRows(eventRows, minuteRows);
    downloadCSV("BoutScout_minute_predictions_final.csv", minuteRows, ["source_file", "sensor_type", "date", "datetime", "minute_of_day", "egg_temperature", "ambient_temperature", "raw_label", "final_label"]);
    downloadCSV("BoutScout_events_final.csv", eventRows, ["source_file", "sensor_type", "date", "event_id", "event_number", "label", "start_minute", "end_minute", "duration_min", "mean_egg_temp", "min_egg_temp", "max_egg_temp", "mean_ambient_temp", "min_ambient_temp", "max_ambient_temp", "min_egg_temp_off", "min_ambient_temp_off", "mean_egg_temp_off", "mean_ambient_temp_off", "mean_egg_temp_on", "mean_ambient_temp_on", "mean_egg_temp_nocturnal", "mean_ambient_temp_nocturnal"]);
    downloadCSV("BoutScout_daily_summary.csv", dailyRows, ["source_file", "sensor_type", "date", "nest_attentiveness_day", "duracion_total_on", "duracion_intervalo_day", "inicio_intervalo", "fin_intervalo", "total_on_full_day", "total_off_full_day", "total_nocturnal_full_day", "avg_duracion_on", "avg_duracion_off", "avg_duracion_nocturnal", "min_duracion_on", "max_duracion_on", "min_duracion_off", "max_duracion_off", "n_eventos_on", "n_eventos_off", "n_eventos_nocturnal", "avg_temp_ambient_diaria", "std_temp_ambient_diaria", "avg_temp_sensor_diaria", "min_temp_sensor_diaria", "max_temp_sensor_diaria", "mean_min_egg_temp_off_bouts", "lowest_egg_temp_recorded_during_off_bout", "mean_min_ambient_temp_off_bouts", "lowest_ambient_temp_recorded_during_off_bout"]);
}

function finishInference() {
    if (analyses.size === 0) {
        alert("No hay datos para exportar.");
        return;
    }
    exportAllResults();
    document.getElementById("thankYouModal").classList.remove("hidden");
}

function closeModal() {
    document.getElementById("thankYouModal").classList.add("hidden");
}

function syncBoutScoutLanguageUI() {
    document.getElementById("btnEditOnText").innerText = isEditMode ? t("btnEditOff") : t("btnEditOn");
    refreshEditControls();
}

Object.assign(window, {
    loadDemoData,
    toggleEditMode,
    startNewLabel,
    applySelectedLabel,
    cancelEditSelection,
    previewBoundaryChange,
    applyBoundaryChange,
    updateLabel,
    finishInference,
    closeModal,
    syncBoutScoutLanguageUI
});
