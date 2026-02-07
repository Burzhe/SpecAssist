(() => {
  const DB_NAME = "specassist_offline";
  const DB_VERSION = 1;
  const STORE_ITEMS = "items";
  const STORE_META = "meta";

  const CATEGORY_STEMS = {
    "шкаф": ["шкаф", "пенал", "гардероб", "купе"],
    "стеллаж": ["стеллаж", "стелаж", "стелл"],
    "кухня": ["кухн"],
    "стол": ["стол", "столешн", "столик"],
    "бенч-стол": ["бенч", "bench", "бенч-стол"],
    "бар": ["бар", "барн", "стойк"],
    "дверь": ["двер", "дверн"],
    "перила": ["перил", "поручн"],
    "зеркало": ["зеркал"],
  };

  const FLAG_LABELS = {
    has_led: "LED",
    mat_ldsp: "ЛДСП",
    mat_mdf: "МДФ",
    mat_veneer: "Шпон",
    has_glass: "Стекло",
    has_metal: "Металл",
  };

  const state = {
    items: [],
    index: null,
    lastResults: [],
    sortKey: "price_unit_ex_vat",
    sortDir: "asc",
    worker: null,
    progress: {
      sheetsTotal: 0,
      sheetsDone: 0,
      rowsTotal: 0,
      rowsInserted: 0,
      rowsSkipped: 0,
    },
  };

  const elements = {
    uploadScreen: document.getElementById("upload-screen"),
    searchScreen: document.getElementById("search-screen"),
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    fileMeta: document.getElementById("file-meta"),
    sheetOptions: document.getElementById("sheet-options"),
    sheetList: document.getElementById("sheet-list"),
    selectAllBtn: document.getElementById("select-all-btn"),
    selectNoneBtn: document.getElementById("select-none-btn"),
    importBtn: document.getElementById("import-btn"),
    progressContainer: document.getElementById("progress-container"),
    overallProgress: document.getElementById("overall-progress"),
    overallProgressLabel: document.getElementById("overall-progress-label"),
    sheetProgress: document.getElementById("sheet-progress"),
    progressStats: document.getElementById("progress-stats"),
    searchInput: document.getElementById("search-input"),
    searchBtn: document.getElementById("search-btn"),
    categoryFilter: document.getElementById("category-filter"),
    flagFilters: document.getElementById("flag-filters"),
    resultsTableBody: document.querySelector("#results-table tbody"),
    resultsSummary: document.getElementById("results-summary"),
    detailsDrawer: document.getElementById("details-drawer"),
    detailsContent: document.getElementById("details-content"),
    closeDrawer: document.getElementById("close-drawer"),
    resetBtn: document.getElementById("reset-btn"),
  };

  const dimInputs = {
    wMin: document.getElementById("w-min"),
    wMax: document.getElementById("w-max"),
    wTol: document.getElementById("w-tol"),
    dMin: document.getElementById("d-min"),
    dMax: document.getElementById("d-max"),
    dTol: document.getElementById("d-tol"),
    hMin: document.getElementById("h-min"),
    hMax: document.getElementById("h-max"),
    hTol: document.getElementById("h-tol"),
  };

  const priceInputs = {
    min: document.getElementById("price-min"),
    max: document.getElementById("price-max"),
  };

  function openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onerror = () => reject(request.error);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE_ITEMS)) {
          db.createObjectStore(STORE_ITEMS, { keyPath: "id" });
        }
        if (!db.objectStoreNames.contains(STORE_META)) {
          db.createObjectStore(STORE_META, { keyPath: "key" });
        }
      };
      request.onsuccess = () => resolve(request.result);
    });
  }

  async function clearDB() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction([STORE_ITEMS, STORE_META], "readwrite");
      tx.objectStore(STORE_ITEMS).clear();
      tx.objectStore(STORE_META).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function saveMeta(key, value) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_META, "readwrite");
      tx.objectStore(STORE_META).put({ key, value });
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function loadMeta(key) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_META, "readonly");
      const req = tx.objectStore(STORE_META).get(key);
      req.onsuccess = () => resolve(req.result ? req.result.value : null);
      req.onerror = () => reject(req.error);
    });
  }

  async function addItems(items) {
    if (!items.length) return;
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_ITEMS, "readwrite");
      const store = tx.objectStore(STORE_ITEMS);
      items.forEach((item) => store.put(item));
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async function loadAllItems() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_ITEMS, "readonly");
      const req = tx.objectStore(STORE_ITEMS).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  function buildIndex(items) {
    const index = new FlexSearch.Index({
      tokenize: "forward",
      cache: true,
    });
    items.forEach((item) => {
      const text = `${item.name || ""} ${item.description || ""}`.toLowerCase();
      index.add(item.id, text);
    });
    state.index = index;
  }

  function computeDerived(item) {
    const widthM = item.w_mm ? item.w_mm / 1000 : null;
    const heightM = item.h_mm ? item.h_mm / 1000 : null;
    item.price_per_lm = widthM ? item.price_unit_ex_vat / widthM : null;
    item.price_per_m2 = widthM && heightM ? item.price_unit_ex_vat / (widthM * heightM) : null;
  }

  function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return Number(value).toLocaleString("ru-RU", { maximumFractionDigits: digits });
  }

  function showScreen(screen) {
    elements.uploadScreen.classList.toggle("active", screen === "upload");
    elements.searchScreen.classList.toggle("active", screen === "search");
  }

  function setupFlagFilters() {
    elements.flagFilters.innerHTML = "";
    Object.entries(FLAG_LABELS).forEach(([key, label]) => {
      const wrapper = document.createElement("div");
      wrapper.className = "flag-item";
      wrapper.innerHTML = `
        <span>${label}</span>
        <select data-flag="${key}">
          <option value="">Любой</option>
          <option value="yes">Должен быть</option>
          <option value="no">Не должен</option>
        </select>
      `;
      elements.flagFilters.appendChild(wrapper);
    });
  }

  function updateCategoryFilter() {
    const categories = new Set(state.items.map((item) => item.category).filter(Boolean));
    elements.categoryFilter.innerHTML = `<option value="">Любая</option>`;
    Array.from(categories)
      .sort()
      .forEach((category) => {
        const option = document.createElement("option");
        option.value = category;
        option.textContent = category;
        elements.categoryFilter.appendChild(option);
      });
  }

  function getSelectedSheets() {
    const selected = [];
    elements.sheetList.querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
      if (checkbox.checked) selected.push(checkbox.value);
    });
    return selected;
  }

  function resetProgress() {
    state.progress = {
      sheetsTotal: 0,
      sheetsDone: 0,
      rowsTotal: 0,
      rowsInserted: 0,
      rowsSkipped: 0,
    };
    elements.sheetProgress.innerHTML = "";
    elements.progressStats.textContent = "";
    elements.overallProgress.value = 0;
    elements.overallProgressLabel.textContent = "0%";
  }

  function updateProgressUI(payload) {
    const { sheetIndex, sheetName, rowsTotal, rowsInserted, rowsSkipped, sheetsTotal } = payload;
    state.progress.sheetsTotal = sheetsTotal;
    state.progress.sheetsDone = sheetIndex + 1;
    state.progress.rowsTotal += rowsTotal;
    state.progress.rowsInserted += rowsInserted;
    state.progress.rowsSkipped += rowsSkipped;

    const progressPercent = Math.round((state.progress.sheetsDone / sheetsTotal) * 100);
    elements.overallProgress.value = progressPercent;
    elements.overallProgressLabel.textContent = `${progressPercent}%`;

    const row = document.createElement("div");
    row.className = "progress-row";
    row.innerHTML = `
      <span>${sheetName}</span>
      <progress value="${rowsInserted}" max="${Math.max(rowsTotal, rowsInserted, 1)}"></progress>
      <span>${rowsInserted}/${rowsTotal}</span>
    `;
    elements.sheetProgress.appendChild(row);

    elements.progressStats.innerHTML = `
      <div>Строк просканировано: <strong>${state.progress.rowsTotal}</strong></div>
      <div>Вставлено: <strong>${state.progress.rowsInserted}</strong></div>
      <div>Пропущено: <strong>${state.progress.rowsSkipped}</strong></div>
      <div>Листов: <strong>${state.progress.sheetsDone}/${state.progress.sheetsTotal}</strong></div>
    `;
  }

  function getFilterValues() {
    const flagValues = {};
    elements.flagFilters.querySelectorAll("select[data-flag]").forEach((select) => {
      const flag = select.dataset.flag;
      if (select.value === "yes") flagValues[flag] = true;
      if (select.value === "no") flagValues[flag] = false;
    });

    const dims = {
      w: { min: parseFloat(dimInputs.wMin.value), max: parseFloat(dimInputs.wMax.value), tol: parseFloat(dimInputs.wTol.value) },
      d: { min: parseFloat(dimInputs.dMin.value), max: parseFloat(dimInputs.dMax.value), tol: parseFloat(dimInputs.dTol.value) },
      h: { min: parseFloat(dimInputs.hMin.value), max: parseFloat(dimInputs.hMax.value), tol: parseFloat(dimInputs.hTol.value) },
    };

    const price = {
      min: parseFloat(priceInputs.min.value),
      max: parseFloat(priceInputs.max.value),
    };

    return {
      query: elements.searchInput.value.trim(),
      category: elements.categoryFilter.value,
      flags: flagValues,
      dims,
      price,
    };
  }

  function withinRange(value, min, max, tol) {
    if (value === null || value === undefined) return false;
    const minVal = Number.isFinite(min) ? min - (Number.isFinite(tol) ? tol : 0) : null;
    const maxVal = Number.isFinite(max) ? max + (Number.isFinite(tol) ? tol : 0) : null;
    if (minVal !== null && value < minVal) return false;
    if (maxVal !== null && value > maxVal) return false;
    return true;
  }

  function applyFilters(items, filters) {
    return items.filter((item) => {
      if (filters.category && item.category !== filters.category) return false;
      for (const [flag, required] of Object.entries(filters.flags)) {
        if (required === true && !item[flag]) return false;
        if (required === false && item[flag]) return false;
      }

      const dimsMap = {
        w: "w_mm",
        d: "d_mm",
        h: "h_mm",
      };
      for (const [key, config] of Object.entries(filters.dims)) {
        const min = config.min;
        const max = config.max;
        const tol = config.tol;
        if (Number.isFinite(min) || Number.isFinite(max)) {
          const value = item[dimsMap[key]];
          if (!withinRange(value, min, max, tol)) return false;
        }
      }

      if (Number.isFinite(filters.price.min) && item.price_unit_ex_vat < filters.price.min) return false;
      if (Number.isFinite(filters.price.max) && item.price_unit_ex_vat > filters.price.max) return false;
      return true;
    });
  }

  function sortResults(items) {
    const direction = state.sortDir === "asc" ? 1 : -1;
    const key = state.sortKey;
    return [...items].sort((a, b) => {
      const aVal = key === "dims" ? `${a.w_mm || ""}x${a.d_mm || ""}x${a.h_mm || ""}` : a[key];
      const bVal = key === "dims" ? `${b.w_mm || ""}x${b.d_mm || ""}x${b.h_mm || ""}` : b[key];
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      if (typeof aVal === "string") return aVal.localeCompare(String(bVal)) * direction;
      return (aVal - bVal) * direction;
    });
  }

  function renderResults(items) {
    state.lastResults = items;
    elements.resultsTableBody.innerHTML = "";
    elements.resultsSummary.textContent = `Найдено: ${items.length}`;
    items.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.name || ""}</td>
        <td>${item.category || "—"}</td>
        <td>${[item.w_mm, item.d_mm, item.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ")}</td>
        <td>${formatNumber(item.price_unit_ex_vat)}</td>
        <td>${formatNumber(item.price_per_lm)}</td>
        <td>${formatNumber(item.price_per_m2)}</td>
        <td>${renderFlagPills(item)}</td>
        <td>${item.source_sheet || ""}</td>
        <td>${item.source_row || ""}</td>
      `;
      tr.addEventListener("click", () => showDetails(item));
      elements.resultsTableBody.appendChild(tr);
    });
  }

  function renderFlagPills(item) {
    return Object.keys(FLAG_LABELS)
      .filter((flag) => item[flag])
      .map((flag) => `<span class="flag-pill">${FLAG_LABELS[flag]}</span>`)
      .join("");
  }

  function normalizeText(text) {
    return text
      .toLowerCase()
      .replace(/ё/g, "е")
      .replace(/[-–—]+/g, " ")
      .replace(/[^\w\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function tokenize(text) {
    return text.match(/[a-zа-я0-9]+/gi) || [];
  }

  function extractCategory(text) {
    const tokens = tokenize(text);
    let bestCategory = null;
    let bestScore = 0;
    for (const [category, stems] of Object.entries(CATEGORY_STEMS)) {
      let score = 0;
      for (const token of tokens) {
        for (const stem of stems) {
          if (token.startsWith(stem)) score += stem.length;
        }
      }
      if (score > bestScore) {
        bestCategory = category;
        bestScore = score;
      }
    }
    return bestCategory;
  }

  function findSimilar(target) {
    const normalized = normalizeText(target.name || "");
    const category = extractCategory(normalized);
    const dims = [target.w_mm, target.d_mm, target.h_mm];

    const candidates = state.items.filter((item) => item.id !== target.id);
    return candidates
      .map((item) => {
        let score = 0;
        if (category && item.category === category) score -= 10;
        let dimScore = 0;
        let dimHits = 0;
        ["w_mm", "d_mm", "h_mm"].forEach((key, idx) => {
          if (dims[idx] && item[key]) {
            dimScore += Math.abs(item[key] - dims[idx]);
            dimHits += 1;
          }
        });
        const materialHits = Object.keys(FLAG_LABELS).reduce((sum, flag) => {
          return sum + (item[flag] && target[flag] ? 1 : 0);
        }, 0);
        score += dimHits ? dimScore : 1e6;
        score -= materialHits * 5;
        return { item, score };
      })
      .sort((a, b) => a.score - b.score)
      .slice(0, 10)
      .map((entry) => entry.item);
  }

  function showDetails(item) {
    const similar = findSimilar(item);
    elements.detailsContent.innerHTML = `
      <div class="details-section">
        <h3>${item.name || ""}</h3>
        <p>${item.description || ""}</p>
      </div>
      <div class="details-section">
        <strong>Параметры</strong>
        <div>Размеры: ${[item.w_mm, item.d_mm, item.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ")}</div>
        <div>Количество: ${formatNumber(item.qty)}</div>
        <div>Цена/ед: ${formatNumber(item.price_unit_ex_vat)}</div>
        <div>Цена/п.м.: ${formatNumber(item.price_per_lm)}</div>
        <div>Цена/м²: ${formatNumber(item.price_per_m2)}</div>
        <div>Категория: ${item.category || "—"}</div>
        <div>Источник: ${item.source_sheet || ""} / строка ${item.source_row || ""}</div>
      </div>
      <div class="details-section">
        <strong>Флаги</strong>
        <div>${renderFlagPills(item) || "—"}</div>
      </div>
      <div class="details-section">
        <strong>Raw</strong>
        <pre>${JSON.stringify(item.raw || {}, null, 2)}</pre>
      </div>
      <div class="details-section">
        <strong>Похожие позиции</strong>
        <ul class="similar-list">
          ${similar
            .map(
              (sim) => `
              <li>
                <div><strong>${sim.name || ""}</strong></div>
                <div>${[sim.w_mm, sim.d_mm, sim.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ")}</div>
                <div>Цена/ед: ${formatNumber(sim.price_unit_ex_vat)}</div>
              </li>
            `,
            )
            .join("")}
        </ul>
      </div>
    `;
    elements.detailsDrawer.classList.add("open");
  }

  function hideDetails() {
    elements.detailsDrawer.classList.remove("open");
  }

  async function handleSearch() {
    const filters = getFilterValues();
    let items = state.items;
    if (filters.query) {
      const ids = state.index.search(filters.query.toLowerCase(), { limit: 5000 });
      const idSet = new Set(ids);
      items = items.filter((item) => idSet.has(item.id));
    }
    items = applyFilters(items, filters);
    items = sortResults(items);
    renderResults(items);
  }

  function createWorker() {
    const workerSource = document.getElementById("worker-src").textContent;
    const vendorUrl = new URL("vendor/xlsx.full.min.js", window.location.href).href;
    const resolvedSource = workerSource.replace("vendor/xlsx.full.min.js", vendorUrl);
    const blob = new Blob([resolvedSource], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    return new Worker(url);
  }

  function updateFileMeta(file, sheetNames) {
    elements.fileMeta.innerHTML = `
      <strong>${file.name}</strong><br/>
      Размер: ${(file.size / (1024 * 1024)).toFixed(2)} MB<br/>
      Листов: ${sheetNames.length}
    `;
    elements.fileMeta.classList.remove("hidden");
  }

  function renderSheetList(sheetNames) {
    elements.sheetList.innerHTML = "";
    sheetNames.forEach((name) => {
      const wrapper = document.createElement("label");
      wrapper.className = "sheet-item";
      wrapper.innerHTML = `<input type="checkbox" value="${name}" checked /> ${name}`;
      elements.sheetList.appendChild(wrapper);
    });
    elements.sheetOptions.classList.remove("hidden");
    elements.importBtn.disabled = false;
  }

  async function importWorkbook(file, sheetNames) {
    resetProgress();
    elements.progressContainer.classList.remove("hidden");
    const arrayBuffer = await file.arrayBuffer();
    const worker = createWorker();
    state.worker = worker;
    state.items = [];

    worker.onmessage = async (event) => {
      const { type, payload } = event.data;
      if (type === "items") {
        payload.items.forEach((item) => computeDerived(item));
        state.items.push(...payload.items);
        await addItems(payload.items);
      }
      if (type === "progress") {
        updateProgressUI(payload);
      }
      if (type === "done") {
        await saveMeta("summary", payload.summary);
        await saveMeta("sheetReports", payload.sheetReports);
        await saveMeta("importedAt", new Date().toISOString());
        buildIndex(state.items);
        updateCategoryFilter();
        showScreen("search");
        await handleSearch();
        worker.terminate();
      }
    };
    worker.onerror = (event) => {
      elements.progressStats.innerHTML = `
        <div>Ошибка импорта: ${event.message || "Не удалось загрузить скрипт обработки."}</div>
        <div>Проверьте, что файлы vendor/xlsx.full.min.js доступны рядом с index.html.</div>
      `;
      elements.overallProgressLabel.textContent = "Ошибка";
      worker.terminate();
    };

    const selectedSheets = getSelectedSheets();
    worker.postMessage({
      type: "start",
      payload: {
        arrayBuffer,
        fileName: file.name,
        sheetNames,
        selectedSheets,
      },
    });
  }

  async function initFromCache() {
    const items = await loadAllItems();
    if (!items.length) return false;
    items.forEach((item) => computeDerived(item));
    state.items = items;
    buildIndex(items);
    updateCategoryFilter();
    showScreen("search");
    await handleSearch();
    return true;
  }

  function setupSorting() {
    document.querySelectorAll("#results-table th[data-sort]").forEach((th) => {
      th.addEventListener("click", () => {
        const key = th.dataset.sort;
        if (state.sortKey === key) {
          state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = key;
          state.sortDir = "asc";
        }
        renderResults(sortResults(state.lastResults));
      });
    });
  }

  function setupEventListeners() {
    elements.dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      elements.dropZone.classList.add("dragover");
    });
    elements.dropZone.addEventListener("dragleave", () => {
      elements.dropZone.classList.remove("dragover");
    });
    elements.dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("dragover");
      const file = event.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    elements.fileInput.addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (file) handleFile(file);
    });

    elements.selectAllBtn.addEventListener("click", () => {
      elements.sheetList.querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
        checkbox.checked = true;
      });
    });

    elements.selectNoneBtn.addEventListener("click", () => {
      elements.sheetList.querySelectorAll("input[type='checkbox']").forEach((checkbox) => {
        checkbox.checked = false;
      });
    });

    elements.importBtn.addEventListener("click", async () => {
      const file = elements.fileInput.files[0];
      if (!file) return;
      const sheetNames = Array.from(elements.sheetList.querySelectorAll("input[type='checkbox']")).map((checkbox) => checkbox.value);
      await clearDB();
      importWorkbook(file, sheetNames);
    });

    elements.searchBtn.addEventListener("click", handleSearch);
    elements.searchInput.addEventListener("keyup", (event) => {
      if (event.key === "Enter") handleSearch();
    });

    Object.values(dimInputs).forEach((input) => input.addEventListener("change", handleSearch));
    Object.values(priceInputs).forEach((input) => input.addEventListener("change", handleSearch));
    elements.categoryFilter.addEventListener("change", handleSearch);
    elements.flagFilters.addEventListener("change", handleSearch);
    elements.closeDrawer.addEventListener("click", hideDetails);

    elements.resetBtn.addEventListener("click", async () => {
      await clearDB();
      state.items = [];
      state.index = null;
      elements.fileInput.value = "";
      elements.sheetOptions.classList.add("hidden");
      elements.fileMeta.classList.add("hidden");
      elements.progressContainer.classList.add("hidden");
      resetProgress();
      showScreen("upload");
    });
  }

  async function handleFile(file) {
    if (!file.name.endsWith(".xlsx")) {
      alert("Пожалуйста, выберите файл .xlsx");
      return;
    }
    updateFileMeta(file, []);
    elements.sheetOptions.classList.add("hidden");
    elements.importBtn.disabled = true;

    const arrayBuffer = await file.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: "array" });
    const sheetNames = workbook.SheetNames;
    updateFileMeta(file, sheetNames);
    renderSheetList(sheetNames);
  }

  async function init() {
    setupFlagFilters();
    setupEventListeners();
    setupSorting();
    const hasCache = await initFromCache();
    if (!hasCache) showScreen("upload");
  }

  init();
})();
