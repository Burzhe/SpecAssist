(() => {
  const DB_NAME = "specassist_offline";
  const DB_VERSION = 1;
  const STORE_ITEMS = "items";
  const STORE_META = "meta";

  const CATEGORY_STEMS = {
    "Шкафы": ["шкаф", "пенал", "гардероб", "купе", "встроен"],
    "Стеллажи": ["стеллаж", "стелл", "этажерк"],
    "Кухни": ["кухн", "столешниц", "фартук"],
    "Столы": ["стол", "столик", "парт"],
    "Кресла и стулья": ["кресл", "стул", "табурет", "пуф"],
    "Бары и стойки": ["бар", "стойк", "ресепшн"],
    "Двери": ["двер", "дверн", "полотн"],
    "Перегородки": ["перегород", "перил", "поручн"],
    "Зеркала": ["зеркал"],
    "Освещение": ["светильн", "люстр", "бра", "торшер", "подсветк"],
    "Мягкая мебель": ["диван", "кровать", "матрас"],
    "Офисная мебель": ["офисн", "рабоч.*мест", "кабинет"],
    "Детская мебель": ["детск", "подростк"],
    "Фурнитура": ["ручк", "петл", "направляющ", "фурнитур"],
    "Прочее": [],
  };

  const FLAG_LABELS = {
    mat_ldsp: "ЛДСП",
    mat_mdf: "МДФ",
    mat_veneer: "Шпон",
    has_glass: "Стекло",
    has_metal: "Металл",
    has_led: "LED",
    has_stone: "Камень",
    has_acrylic: "Акрил",
  };

  const MAPPING_FIELDS = [
    { key: "name_col", label: "Наименование" },
    { key: "dims_col", label: "Размеры (Ш×Г×В)" },
    { key: "desc_col", label: "Описание / Материал" },
    { key: "qty_col", label: "Количество" },
    { key: "price_unit_col", label: "Цена за ед." },
    { key: "total_col", label: "Итого" },
  ];

  const MAPPING_STORAGE_KEY = "specassist_custom_mappings";
  const TEMPLATE_STORAGE_KEY = "specassist_mapping_templates";
  const FILTER_STORAGE_KEY = "specassist_last_search";

  const state = {
    items: [],
    index: null,
    lastResults: [],
    sortKey: "price_unit_ex_vat",
    sortDir: "asc",
    viewMode: "table",
    compareIds: new Set(),
    worker: null,
    searchTimer: null,
    filterTimer: null,
    previewMeta: {},
    workbook: null,
    sheetNames: [],
    customMappings: { sheets: {}, global: null },
    mappingTemplates: [],
    activeMappingSheet: null,
    scrollTimer: null,
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
    mappingBtn: document.getElementById("mapping-btn"),
    importBtn: document.getElementById("import-btn"),
    progressContainer: document.getElementById("progress-container"),
    overallProgress: document.getElementById("overall-progress"),
    overallProgressLabel: document.getElementById("overall-progress-label"),
    progressMessage: document.getElementById("progress-message"),
    sheetProgress: document.getElementById("sheet-progress"),
    progressStats: document.getElementById("progress-stats"),
    searchInput: document.getElementById("search-input"),
    searchBtn: document.getElementById("search-btn"),
    categoryFilter: document.getElementById("category-filter"),
    flagFilters: document.getElementById("flag-filters"),
    resultsTableBody: document.querySelector("#results-table tbody"),
    resultsSummary: document.getElementById("results-summary"),
    resultsEmpty: document.getElementById("results-empty"),
    resultsLoading: document.getElementById("results-loading"),
    scrollSkeleton: document.getElementById("scroll-skeleton"),
    cardsView: document.getElementById("cards-view"),
    tableWrap: document.getElementById("table-wrap"),
    detailsDrawer: document.getElementById("details-drawer"),
    detailsContent: document.getElementById("details-content"),
    closeDrawer: document.getElementById("close-drawer"),
    resetBtn: document.getElementById("reset-btn"),
    resetFiltersBtn: document.getElementById("reset-filters-btn"),
    activeFilters: document.getElementById("active-filters"),
    categoryCount: document.getElementById("category-count"),
    viewTableBtn: document.getElementById("view-table-btn"),
    viewCardsBtn: document.getElementById("view-cards-btn"),
    exportBtn: document.getElementById("export-btn"),
    increaseTolBtn: document.getElementById("increase-tol-btn"),
    removeLedBtn: document.getElementById("remove-led-btn"),
    sheetPreview: document.getElementById("sheet-preview"),
    sheetPreviewTabs: document.getElementById("sheet-preview-tabs"),
    sheetPreviewContent: document.getElementById("sheet-preview-content"),
    compareBtn: document.getElementById("compare-btn"),
    compareModal: document.getElementById("compare-modal"),
    compareTable: document.getElementById("compare-table"),
    compareSummary: document.getElementById("compare-summary"),
    closeCompare: document.getElementById("close-compare"),
    themeToggle: document.getElementById("theme-toggle"),
    columnMappingModal: document.getElementById("column-mapping-modal"),
    closeColumnMapping: document.getElementById("close-column-mapping"),
    mappingSheetName: document.getElementById("mapping-sheet-name"),
    mappingPreview: document.getElementById("mapping-preview"),
    mappingAutoBtn: document.getElementById("mapping-auto-btn"),
    mappingSaveBtn: document.getElementById("mapping-save-btn"),
    mappingApplyAll: document.getElementById("mapping-apply-all"),
    mappingTemplateSelect: document.getElementById("mapping-template-select"),
    mappingTemplateSave: document.getElementById("mapping-template-save"),
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
    wMinRange: document.getElementById("w-min-range"),
    wMaxRange: document.getElementById("w-max-range"),
    dMinRange: document.getElementById("d-min-range"),
    dMaxRange: document.getElementById("d-max-range"),
    hMinRange: document.getElementById("h-min-range"),
    hMaxRange: document.getElementById("h-max-range"),
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

  function loadCustomMappings() {
    try {
      const raw = localStorage.getItem(MAPPING_STORAGE_KEY);
      if (!raw) return { sheets: {}, global: null };
      const parsed = JSON.parse(raw);
      return {
        sheets: parsed.sheets || {},
        global: parsed.global || null,
      };
    } catch (error) {
      console.warn("[mapping] failed to load custom mappings", error);
      return { sheets: {}, global: null };
    }
  }

  function saveCustomMappings() {
    localStorage.setItem(MAPPING_STORAGE_KEY, JSON.stringify(state.customMappings));
  }

  function loadMappingTemplates() {
    try {
      const raw = localStorage.getItem(TEMPLATE_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      console.warn("[mapping] failed to load templates", error);
      return [];
    }
  }

  function saveMappingTemplates() {
    localStorage.setItem(TEMPLATE_STORAGE_KEY, JSON.stringify(state.mappingTemplates));
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

  function debounce(fn, delay) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  function throttle(fn, delay) {
    let lastCall = 0;
    let timeoutId;
    return (...args) => {
      const now = Date.now();
      const remaining = delay - (now - lastCall);
      if (remaining <= 0) {
        lastCall = now;
        fn(...args);
      } else {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          lastCall = Date.now();
          fn(...args);
        }, remaining);
      }
    };
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
        <span>${label} <span class="pill-count" data-flag-count="${key}"></span></span>
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

  function updateFilterCounts(items) {
    const categoryCounts = items.reduce((acc, item) => {
      if (!item.category) return acc;
      acc[item.category] = (acc[item.category] || 0) + 1;
      return acc;
    }, {});
    const selectedCategory = elements.categoryFilter.value;
    const categoryLabel = selectedCategory ? `${selectedCategory} (${categoryCounts[selectedCategory] || 0})` : `Всего (${items.length})`;
    elements.categoryCount.textContent = categoryLabel;

    Object.keys(FLAG_LABELS).forEach((flag) => {
      const count = items.filter((item) => item[flag]).length;
      const badge = elements.flagFilters.querySelector(`[data-flag-count="${flag}"]`);
      if (badge) badge.textContent = count ? `(${count})` : "";
    });
  }

  function syncRangePair(minRange, maxRange, minInput, maxInput) {
    const minValue = Number(minRange.value);
    const maxValue = Number(maxRange.value);
    if (minValue > maxValue) {
      minRange.value = maxValue;
    }
    minInput.value = minRange.value !== "0" ? minRange.value : "";
    maxInput.value = maxRange.value !== maxRange.max ? maxRange.value : "";
  }

  function setRangeFromInput(input, range, fallbackValue) {
    const value = parseFloat(input.value);
    if (Number.isFinite(value)) {
      range.value = Math.min(Math.max(value, Number(range.min)), Number(range.max));
    } else {
      range.value = fallbackValue;
    }
  }

  function updateActiveFilters(filters) {
    let count = 0;
    if (filters.query) count += 1;
    if (filters.category) count += 1;
    count += Object.keys(filters.flags).length;
    ["w", "d", "h"].forEach((key) => {
      const dim = filters.dims[key];
      if (Number.isFinite(dim.min) || Number.isFinite(dim.max)) count += 1;
      if (Number.isFinite(dim.tol) && dim.tol > 0) count += 1;
    });
    if (Number.isFinite(filters.price.min) || Number.isFinite(filters.price.max)) count += 1;

    elements.activeFilters.textContent = count ? `Применено фильтров: ${count}` : "Фильтры не применены";
    elements.resetFiltersBtn.classList.toggle("hidden", count === 0);
  }

  function logSearchDiagnostics(stage, payload) {
    if (!payload) return;
    console.info(`[search] ${stage}`, payload);
  }

  function updateEmptyState({ queryItems, filters, results }) {
    if (results.length) {
      elements.increaseTolBtn.disabled = false;
      elements.removeLedBtn.disabled = false;
      return;
    }
    const hints = [];
    const dimsActive = Object.values(filters.dims).some((dim) => Number.isFinite(dim.min) || Number.isFinite(dim.max));
    if (dimsActive) hints.push("Попробуйте увеличить толерантность размеров.");
    if (filters.flags.has_led === true) {
      const relaxedFlags = { ...filters.flags };
      delete relaxedFlags.has_led;
      const relaxedItems = applyFilters(queryItems, { ...filters, flags: relaxedFlags });
      if (relaxedItems.length) {
        hints.push(`Найдено 0 с LED, но ${relaxedItems.length} без подсветки — убрать фильтр?`);
      }
    }
    elements.increaseTolBtn.disabled = !dimsActive;
    elements.removeLedBtn.disabled = filters.flags.has_led !== true;
    const hintText = hints.length ? hints.join(" ") : "Попробуйте изменить запрос или снять часть фильтров.";
    const hintEl = elements.resultsEmpty.querySelector("[data-empty-hint]");
    if (hintEl) hintEl.textContent = hintText;
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
    elements.progressMessage.textContent = "";
  }

  function updateProgressUI(payload) {
    const {
      sheetIndex,
      sheetName,
      rowsTotal,
      rowsInserted,
      rowsSkipped,
      sheetsTotal,
      summary,
    } = payload;
    state.progress.sheetsTotal = sheetsTotal;
    state.progress.sheetsDone = sheetIndex + 1;
    if (summary) {
      state.progress.rowsTotal = summary.rows_total;
      state.progress.rowsInserted = summary.rows_inserted;
      state.progress.rowsSkipped = summary.rows_skipped;
    }

    const progressPercent = Math.round((state.progress.sheetsDone / sheetsTotal) * 100);
    elements.overallProgress.value = progressPercent;
    elements.overallProgressLabel.textContent = `${progressPercent}%`;
    if (progressPercent < 30) elements.progressMessage.textContent = "🔍 Сканируем листы...";
    else if (progressPercent < 80) elements.progressMessage.textContent = "📊 Индексируем данные...";
    else elements.progressMessage.textContent = "✨ Готово!";

    let row = elements.sheetProgress.querySelector(`[data-sheet="${sheetName}"]`);
    if (!row) {
      row = document.createElement("div");
      row.className = "progress-row";
      row.dataset.sheet = sheetName;
      row.innerHTML = `
        <span>${sheetName}</span>
        <progress value="0" max="1"></progress>
        <span>0/0</span>
      `;
      elements.sheetProgress.appendChild(row);
    }
    const progressEl = row.querySelector("progress");
    const labelEl = row.querySelector("span:last-child");
    progressEl.value = rowsInserted;
    progressEl.max = Math.max(rowsTotal, rowsInserted, 1);
    labelEl.textContent = `${rowsInserted}/${rowsTotal}`;

    elements.progressStats.innerHTML = `
      <div>Строк просканировано: <strong>${state.progress.rowsTotal}</strong></div>
      <div>Вставлено: <strong>${state.progress.rowsInserted}</strong></div>
      <div>Пропущено: <strong>${state.progress.rowsSkipped}</strong></div>
      <div>Листов: <strong>${state.progress.sheetsDone}/${state.progress.sheetsTotal}</strong></div>
    `;
  }

  function resetFiltersToDefault() {
    elements.searchInput.value = "";
    elements.categoryFilter.value = "";
    elements.flagFilters.querySelectorAll("select[data-flag]").forEach((select) => (select.value = ""));
    Object.values(priceInputs).forEach((input) => (input.value = ""));
    Object.entries(dimInputs).forEach(([key, input]) => {
      if (key.endsWith("Range")) return;
      input.value = "";
    });
    dimInputs.wMinRange.value = 0;
    dimInputs.wMaxRange.value = 6000;
    dimInputs.dMinRange.value = 0;
    dimInputs.dMaxRange.value = 6000;
    dimInputs.hMinRange.value = 0;
    dimInputs.hMaxRange.value = 6000;
  }

  function getColumnLetter(index) {
    let result = "";
    let current = index + 1;
    while (current > 0) {
      const remainder = (current - 1) % 26;
      result = String.fromCharCode(65 + remainder) + result;
      current = Math.floor((current - 1) / 26);
    }
    return result;
  }

  function buildColumnOptions(maxCols) {
    const options = [{ value: "", label: "—" }];
    for (let i = 0; i < maxCols; i += 1) {
      const letter = getColumnLetter(i);
      options.push({ value: String(i), label: `${letter} (${i + 1})` });
    }
    return options;
  }

  function analyzeColumnContent(values) {
    const samples = values.filter((v) => v != null).slice(0, 20);
    if (!samples.length) return { type: null, confidence: 0 };

    let dimMatches = 0;
    let priceMatches = 0;
    let nameMatches = 0;

    samples.forEach((val) => {
      const str = String(val).toLowerCase();
      if (/\d{2,4}\s*[x×*]\s*\d{2,4}/.test(str)) dimMatches += 1;
      if (/^\d{1,8}([.,]\d{1,2})?$/.test(str) && parseFloat(str) > 100) priceMatches += 1;
      if (str.length > 10 && /[а-я]{3,}/.test(str)) nameMatches += 1;
    });

    const total = samples.length;
    if (dimMatches / total > 0.5) return { type: "dims", confidence: dimMatches / total };
    if (priceMatches / total > 0.6) return { type: "price", confidence: priceMatches / total };
    if (nameMatches / total > 0.7) return { type: "name", confidence: nameMatches / total };

    return { type: null, confidence: 0 };
  }

  function suggestMapping(headers, sampleRows = []) {
    const suggestions = {};
    headers.forEach((header, idx) => {
      const normalized = normalizeText(String(header || ""));
      if (!normalized) return;
      if (suggestions.name_col === undefined && /(наимен|товар|номенклат|издел|позици)/.test(normalized)) {
        suggestions.name_col = idx;
      }
      if (suggestions.dims_col === undefined && /(размер|габарит|шхгхв|шир|выс|глуб)/.test(normalized)) {
        suggestions.dims_col = idx;
      }
      if (suggestions.desc_col === undefined && /(описан|материал|коммент|примечан)/.test(normalized)) {
        suggestions.desc_col = idx;
      }
      if (suggestions.qty_col === undefined && /(колич|кол-?во|кол\b|шт)/.test(normalized)) {
        suggestions.qty_col = idx;
      }
      if (suggestions.price_unit_col === undefined && /(цена|стоим|руб).*(ед|шт)|цена.*ед|за\s*ед/.test(normalized)) {
        suggestions.price_unit_col = idx;
      }
      if (suggestions.total_col === undefined && /(итого|сумма|всего)/.test(normalized)) {
        suggestions.total_col = idx;
      }
    });

    if (sampleRows.length) {
      const maxCols = Math.max(
        headers.length,
        ...sampleRows.map((row) => (row ? row.length : 0)),
      );
      const usedCols = new Set(Object.values(suggestions).filter((value) => value !== undefined));
      for (let colIdx = 0; colIdx < maxCols; colIdx += 1) {
        if (usedCols.has(colIdx)) continue;
        const values = sampleRows.map((row) => (row ? row[colIdx] : null));
        const analysis = analyzeColumnContent(values);
        if (analysis.type === "name" && suggestions.name_col === undefined) {
          suggestions.name_col = colIdx;
          usedCols.add(colIdx);
        }
        if (analysis.type === "dims" && suggestions.dims_col === undefined) {
          suggestions.dims_col = colIdx;
          usedCols.add(colIdx);
        }
        if (analysis.type === "price" && suggestions.price_unit_col === undefined) {
          suggestions.price_unit_col = colIdx;
          usedCols.add(colIdx);
        }
      }
    }
    return suggestions;
  }

  function getModalMappingSelects() {
    return Array.from(elements.columnMappingModal.querySelectorAll("select[data-mapping]"));
  }

  function applyMappingToModal(mapping) {
    getModalMappingSelects().forEach((select) => {
      const key = select.dataset.mapping;
      const value = mapping && mapping[key] !== undefined && mapping[key] !== null ? String(mapping[key]) : "";
      select.value = value;
    });
  }

  function collectMappingFromModal() {
    const mapping = {};
    getModalMappingSelects().forEach((select) => {
      const key = select.dataset.mapping;
      mapping[key] = select.value !== "" ? Number(select.value) : null;
    });
    return mapping;
  }

  function renderMappingPreview(sheetName) {
    if (!state.workbook) return;
    const sheet = state.workbook.Sheets[sheetName];
    if (!sheet) return;
    const rows = XLSX.utils.sheet_to_json(sheet, { header: 1, blankrows: false, raw: true });
    const previewRows = rows.slice(0, 12);
    const meta = state.previewMeta[sheetName] || detectPreviewHeader(previewRows);
    const headerRowIndex = meta.headerRowIndex ?? 0;
    const headerRow = previewRows[headerRowIndex] || [];
    const maxCols = Math.max(1, ...previewRows.map((row) => (row ? row.length : 0)));
    const options = buildColumnOptions(maxCols);
    const currentSelection = collectMappingFromModal();

    getModalMappingSelects().forEach((select) => {
      select.innerHTML = "";
      options.forEach((option) => {
        const opt = document.createElement("option");
        opt.value = option.value;
        opt.textContent = option.label;
        select.appendChild(opt);
      });
    });
    applyMappingToModal(currentSelection);

    const selectedCols = new Set(
      getModalMappingSelects()
        .map((select) => (select.value !== "" ? Number(select.value) : null))
        .filter((value) => value !== null),
    );

    const table = document.createElement("table");
    table.className = "mapping-preview-table";
    const thead = document.createElement("thead");
    const letterRow = document.createElement("tr");
    for (let i = 0; i < maxCols; i += 1) {
      const th = document.createElement("th");
      if (selectedCols.has(i)) th.classList.add("mapping-highlight");
      th.textContent = getColumnLetter(i);
      letterRow.appendChild(th);
    }
    thead.appendChild(letterRow);
    const headerRowEl = document.createElement("tr");
    for (let i = 0; i < maxCols; i += 1) {
      const th = document.createElement("th");
      if (selectedCols.has(i)) th.classList.add("mapping-highlight");
      th.textContent = headerRow[i] ?? "";
      headerRowEl.appendChild(th);
    }
    thead.appendChild(headerRowEl);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    previewRows.slice(headerRowIndex + 1, headerRowIndex + 11).forEach((row) => {
      const tr = document.createElement("tr");
      for (let i = 0; i < maxCols; i += 1) {
        const td = document.createElement("td");
        if (selectedCols.has(i)) td.classList.add("mapping-highlight");
        td.textContent = row?.[i] ?? "";
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    elements.mappingPreview.innerHTML = "";
    elements.mappingPreview.appendChild(table);
  }

  function openColumnMappingModal(sheetName) {
    state.activeMappingSheet = sheetName;
    elements.mappingSheetName.textContent = sheetName;
    const mapping = state.customMappings.sheets[sheetName] || state.customMappings.global || {};
    elements.mappingApplyAll.checked = false;
    elements.mappingTemplateSelect.value = "";
    renderMappingPreview(sheetName);
    applyMappingToModal(mapping);
    renderMappingPreview(sheetName);
    elements.columnMappingModal.classList.remove("hidden");
  }

  function closeColumnMappingModal() {
    elements.columnMappingModal.classList.add("hidden");
  }

  function refreshTemplateSelect() {
    elements.mappingTemplateSelect.innerHTML = `<option value="">Без шаблона</option>`;
    state.mappingTemplates.forEach((template, idx) => {
      const option = document.createElement("option");
      option.value = String(idx);
      option.textContent = template.name;
      elements.mappingTemplateSelect.appendChild(option);
    });
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

  function saveFiltersToLocalStorage(filters) {
    localStorage.setItem(FILTER_STORAGE_KEY, JSON.stringify(filters));
  }

  function loadFiltersFromLocalStorage() {
    const saved = localStorage.getItem(FILTER_STORAGE_KEY);
    if (!saved) return null;
    try {
      return JSON.parse(saved);
    } catch (error) {
      return null;
    }
  }

  function applyFiltersToUI(filters) {
    if (!filters) return;
    elements.searchInput.value = filters.query || "";
    elements.categoryFilter.value = filters.category || "";
    elements.flagFilters.querySelectorAll("select[data-flag]").forEach((select) => {
      const flag = select.dataset.flag;
      if (filters.flags?.[flag] === true) select.value = "yes";
      else if (filters.flags?.[flag] === false) select.value = "no";
      else select.value = "";
    });

    const dims = filters.dims || {};
    const setDimValues = (key, inputMin, inputMax, inputTol, rangeMin, rangeMax) => {
      const config = dims[key] || {};
      inputMin.value = Number.isFinite(config.min) ? config.min : "";
      inputMax.value = Number.isFinite(config.max) ? config.max : "";
      inputTol.value = Number.isFinite(config.tol) ? config.tol : "";
      rangeMin.value = Number.isFinite(config.min) ? config.min : 0;
      rangeMax.value = Number.isFinite(config.max) ? config.max : rangeMax.max;
    };
    setDimValues("w", dimInputs.wMin, dimInputs.wMax, dimInputs.wTol, dimInputs.wMinRange, dimInputs.wMaxRange);
    setDimValues("d", dimInputs.dMin, dimInputs.dMax, dimInputs.dTol, dimInputs.dMinRange, dimInputs.dMaxRange);
    setDimValues("h", dimInputs.hMin, dimInputs.hMax, dimInputs.hTol, dimInputs.hMinRange, dimInputs.hMaxRange);

    priceInputs.min.value = Number.isFinite(filters.price?.min) ? filters.price.min : "";
    priceInputs.max.value = Number.isFinite(filters.price?.max) ? filters.price.max : "";
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

      const dims = filters.dims;

      if (Number.isFinite(dims.w.min) || Number.isFinite(dims.w.max)) {
        const w = item.w_mm;
        if (!w) return false;
        const tol = dims.w.tol || 0;
        if (Number.isFinite(dims.w.min) && w < dims.w.min - tol) return false;
        if (Number.isFinite(dims.w.max) && w > dims.w.max + tol) return false;
      }

      if (Number.isFinite(dims.d.min) || Number.isFinite(dims.d.max)) {
        const d = item.d_mm;
        if (!d) return false;
        const tol = dims.d.tol || 0;
        if (Number.isFinite(dims.d.min) && d < dims.d.min - tol) return false;
        if (Number.isFinite(dims.d.max) && d > dims.d.max + tol) return false;
      }

      if (Number.isFinite(dims.h.min) || Number.isFinite(dims.h.max)) {
        const h = item.h_mm;
        if (!h) return false;
        const tol = dims.h.tol || 0;
        if (Number.isFinite(dims.h.min) && h < dims.h.min - tol) return false;
        if (Number.isFinite(dims.h.max) && h > dims.h.max + tol) return false;
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
    elements.resultsSummary.textContent = `Найдено: ${items.length}`;
    elements.resultsEmpty.classList.toggle("hidden", items.length > 0);
    elements.scrollSkeleton.classList.add("hidden");

    const renderTableSlice = (start, end) => {
      elements.resultsTableBody.innerHTML = "";
      const fragment = document.createDocumentFragment();
      const total = items.length;
      const topSpacer = document.createElement("tr");
      topSpacer.className = "spacer-row";
      topSpacer.innerHTML = `<td colspan="11" style="height:${start * 44}px"></td>`;
      fragment.appendChild(topSpacer);
      items.slice(start, end).forEach((item) => {
        const tr = document.createElement("tr");
        tr.className = "result-row";
        tr.innerHTML = `
          <td><input type="checkbox" data-compare="${item.id}" ${state.compareIds.has(item.id) ? "checked" : ""} /></td>
          <td>${item.name || ""}</td>
          <td>${item.description ? `${item.description.substring(0, 50)}${item.description.length > 50 ? "..." : ""}` : "—"}</td>
          <td>${item.category || "—"}</td>
          <td>${[item.w_mm, item.d_mm, item.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ")}</td>
          <td>${formatNumber(item.price_unit_ex_vat)}</td>
          <td>${formatNumber(item.price_per_lm)}</td>
          <td>${formatNumber(item.price_per_m2)}</td>
          <td>${renderFlagPills(item)}</td>
          <td>${item.source_sheet || ""}</td>
          <td>${item.source_row || ""}</td>
        `;
        tr.addEventListener("click", (event) => {
          if (event.target.matches("input[type='checkbox']")) return;
          showDetails(item);
        });
        fragment.appendChild(tr);
      });
      const bottomSpacer = document.createElement("tr");
      bottomSpacer.className = "spacer-row";
      bottomSpacer.innerHTML = `<td colspan="11" style="height:${Math.max(total - end, 0) * 44}px"></td>`;
      fragment.appendChild(bottomSpacer);
      elements.resultsTableBody.appendChild(fragment);
      elements.resultsTableBody.querySelectorAll("input[data-compare]").forEach((checkbox) => {
        checkbox.addEventListener("change", (event) => {
          const id = Number(event.target.dataset.compare);
          if (event.target.checked) state.compareIds.add(id);
          else state.compareIds.delete(id);
          updateCompareButton();
        });
      });
    };

    const renderCardsSlice = (start, end) => {
      elements.cardsView.innerHTML = "";
      elements.cardsView.style.paddingTop = `${start * 220}px`;
      elements.cardsView.style.paddingBottom = `${Math.max(items.length - end, 0) * 220}px`;
      const fragment = document.createDocumentFragment();
      items.slice(start, end).forEach((item) => {
        const card = document.createElement("div");
        card.className = "card-item";
        card.innerHTML = `
          <div class="cards-actions">
            <strong>${item.name || "Без названия"}</strong>
            <label><input type="checkbox" data-compare="${item.id}" ${state.compareIds.has(item.id) ? "checked" : ""} /> сравнить</label>
          </div>
          <div class="card-description">${item.description ? `${item.description.substring(0, 70)}${item.description.length > 70 ? "..." : ""}` : "—"}</div>
          <div class="dims-icon">📦 ${[item.w_mm, item.d_mm, item.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ")}</div>
          <div>Цена/ед: <strong>${formatNumber(item.price_unit_ex_vat)}</strong></div>
          <div class="card-badges">${renderBadgeChips(item) || ""}</div>
          <div class="cards-actions">
            <button class="ghost" data-details="${item.id}">Подробнее</button>
          </div>
        `;
        fragment.appendChild(card);
      });
      elements.cardsView.appendChild(fragment);
      elements.cardsView.querySelectorAll("button[data-details]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const item = items.find((entry) => entry.id === Number(btn.dataset.details));
          if (item) showDetails(item);
        });
      });
      elements.cardsView.querySelectorAll("input[data-compare]").forEach((checkbox) => {
        checkbox.addEventListener("change", (event) => {
          const id = Number(event.target.dataset.compare);
          if (event.target.checked) state.compareIds.add(id);
          else state.compareIds.delete(id);
          updateCompareButton();
        });
      });
    };

    const triggerScrollSkeleton = () => {
      if (!items.length) return;
      elements.scrollSkeleton.classList.remove("hidden");
      clearTimeout(state.scrollTimer);
      state.scrollTimer = setTimeout(() => {
        elements.scrollSkeleton.classList.add("hidden");
      }, 200);
    };

    const renderWithVirtualization = (isScroll = false) => {
      if (isScroll) triggerScrollSkeleton();
      if (state.viewMode === "table") {
        const rowHeight = 44;
        const visibleCount = Math.ceil(elements.tableWrap.clientHeight / rowHeight) + 10;
        const startIndex = Math.max(Math.floor(elements.tableWrap.scrollTop / rowHeight) - 5, 0);
        renderTableSlice(startIndex, startIndex + visibleCount);
      } else {
        const cardHeight = 220;
        const visibleCount = Math.ceil(elements.cardsView.clientHeight / cardHeight) + 6;
        const startIndex = Math.max(Math.floor(elements.cardsView.scrollTop / cardHeight) - 3, 0);
        renderCardsSlice(startIndex, startIndex + visibleCount);
      }
    };

    renderWithVirtualization();
    elements.tableWrap.onscroll = () => renderWithVirtualization(true);
    elements.cardsView.onscroll = () => renderWithVirtualization(true);
    updateCompareButton();
  }

  function renderFlagPills(item) {
    return Object.keys(FLAG_LABELS)
      .filter((flag) => item[flag])
      .map((flag) => `<span class="flag-pill">${FLAG_LABELS[flag]}</span>`)
      .join("");
  }

  function renderBadgeChips(item) {
    return Object.keys(FLAG_LABELS)
      .filter((flag) => item[flag])
      .map((flag) => `<span class="badge-chip">${FLAG_LABELS[flag]}</span>`)
      .join("");
  }

  function updateCompareButton() {
    const count = state.compareIds.size;
    elements.compareBtn.textContent = `Сравнить (${count})`;
    elements.compareBtn.classList.toggle("hidden", count === 0);
  }

  function renderCompareModal() {
    const compareItems = state.items.filter((item) => state.compareIds.has(item.id));
    if (!compareItems.length) return;
    const prices = compareItems.map((item) => item.price_unit_ex_vat).filter(Number.isFinite);
    const pricesPerLm = compareItems.map((item) => item.price_per_lm).filter(Number.isFinite);
    const pricesPerM2 = compareItems.map((item) => item.price_per_m2).filter(Number.isFinite);
    const minPrice = prices.length ? Math.min(...prices) : null;
    const maxPrice = prices.length ? Math.max(...prices) : null;
    const minPriceM2 = pricesPerM2.length ? Math.min(...pricesPerM2) : null;
    const maxPriceM2 = pricesPerM2.length ? Math.max(...pricesPerM2) : null;
    const avgPriceLm = pricesPerLm.length ? pricesPerLm.reduce((sum, val) => sum + val, 0) / pricesPerLm.length : null;

    const rows = [
      { label: "Название", key: "name" },
      { label: "Категория", key: "category" },
      { label: "Размеры", key: "dims" },
      { label: "Цена/ед", key: "price_unit_ex_vat", highlight: true },
      { label: "Цена/м²", key: "price_per_m2" },
      { label: "Флаги", key: "flags" },
    ];

    elements.compareTable.innerHTML = "";
    rows.forEach((row) => {
      const rowEl = document.createElement("div");
      rowEl.className = "compare-row";
      const label = document.createElement("strong");
      label.textContent = row.label;
      rowEl.appendChild(label);
      compareItems.forEach((item) => {
        const cell = document.createElement("div");
        if (row.key === "dims") {
          cell.textContent = [item.w_mm, item.d_mm, item.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ");
        } else if (row.key === "flags") {
          cell.innerHTML = renderFlagPills(item) || "—";
        } else if (row.key === "price_unit_ex_vat") {
          const price = item.price_unit_ex_vat;
          cell.textContent = formatNumber(price);
          if (row.highlight && Number.isFinite(price) && minPrice !== null && maxPrice !== null) {
            if (price === minPrice) cell.classList.add("highlight-best");
            if (price === maxPrice) cell.classList.add("highlight-worst");
          }
        } else if (row.key === "price_per_m2") {
          const price = item.price_per_m2;
          cell.textContent = formatNumber(price);
          if (Number.isFinite(price) && minPriceM2 !== null && maxPriceM2 !== null) {
            if (price === minPriceM2) cell.classList.add("highlight-best");
            if (price === maxPriceM2) cell.classList.add("highlight-worst");
          }
        } else {
          cell.textContent = item[row.key] || "—";
        }
        rowEl.appendChild(cell);
      });
      elements.compareTable.appendChild(rowEl);
    });
    const diffPercent = minPrice && maxPrice && minPrice > 0 ? ((maxPrice - minPrice) / minPrice) * 100 : null;
    elements.compareSummary.innerHTML = `
      <strong>Итоги:</strong>
      <div>Разница в цене: ${diffPercent !== null ? `${diffPercent.toFixed(1)}%` : "—"}</div>
      <div>Средняя цена/п.м.: ${avgPriceLm !== null ? `${formatNumber(avgPriceLm)} ₽` : "—"}</div>
    `;
    elements.compareModal.classList.remove("hidden");
  }

  function exportItemsToExcel(items, filename) {
    const rows = items.map((item) => ({
      name: item.name,
      description: item.description,
      category: item.category,
      w_mm: item.w_mm,
      d_mm: item.d_mm,
      h_mm: item.h_mm,
      price_unit_ex_vat: item.price_unit_ex_vat,
      price_per_lm: item.price_per_lm,
      price_per_m2: item.price_per_m2,
      flags: Object.keys(FLAG_LABELS)
        .filter((flag) => item[flag])
        .map((flag) => FLAG_LABELS[flag])
        .join(", "),
      source_sheet: item.source_sheet,
      source_row: item.source_row,
    }));
    const sheet = XLSX.utils.json_to_sheet(rows);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, sheet, "results");
    XLSX.writeFile(workbook, filename);
  }

  function setViewMode(mode) {
    state.viewMode = mode;
    elements.viewTableBtn.classList.toggle("active", mode === "table");
    elements.viewCardsBtn.classList.toggle("active", mode === "cards");
    elements.tableWrap.classList.toggle("hidden", mode !== "table");
    elements.cardsView.classList.toggle("hidden", mode !== "cards");
    renderResults(state.lastResults);
  }

  function triggerConfetti() {
    const confetti = document.createElement("div");
    confetti.textContent = "🎉";
    confetti.style.position = "fixed";
    confetti.style.top = "20px";
    confetti.style.right = "20px";
    confetti.style.fontSize = "32px";
    confetti.style.zIndex = "40";
    document.body.appendChild(confetti);
    setTimeout(() => confetti.remove(), 1200);
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
    const tokens = text.toLowerCase().match(/[а-яa-z0-9]+/gi) || [];
    let bestCategory = "Прочее";
    let bestScore = 0;

    for (const [category, stems] of Object.entries(CATEGORY_STEMS)) {
      if (category === "Прочее") continue;
      let score = 0;
      for (const token of tokens) {
        for (const stem of stems) {
          const regex = new RegExp(`^${stem}`, "i");
          if (regex.test(token)) {
            score += stem.length;
          }
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

  function copyItemToClipboard(item) {
    const text = `
${item.name || "—"}
Размеры: ${item.w_mm || "—"}×${item.d_mm || "—"}×${item.h_mm || "—"} мм
Количество: ${item.qty || 1}
Цена/ед: ${formatNumber(item.price_unit_ex_vat)} ₽
Цена/п.м.: ${formatNumber(item.price_per_lm)} ₽
Источник: ${item.source_sheet || "—"}, строка ${item.source_row || "—"}
    `.trim();
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text);
      alert("Скопировано в буфер обмена!");
    }
  }

  function showDetails(item) {
    const similar = findSimilar(item);
    const compareBtnLabel = state.compareIds.has(item.id) ? "Убрать из сравнения" : "Добавить в сравнение";
    elements.detailsContent.innerHTML = `
      <div class="details-section">
        <h3>${item.name || ""}</h3>
        <p>${item.description || ""}</p>
        <div class="details-actions">
          <button id="details-compare-btn" class="ghost">${compareBtnLabel}</button>
          <button id="details-copy-btn" class="ghost">📋 Скопировать</button>
        </div>
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
        <strong>Сравнение цен</strong>
        <canvas id="price-chart" width="320" height="160"></canvas>
      </div>
      <div class="details-section">
        <strong>Raw</strong>
        <pre>${JSON.stringify(item.raw || {}, null, 2)}</pre>
      </div>
      <div class="details-section">
        <strong>Похожие позиции</strong>
        <div class="similar-cards">
          ${similar
            .map(
              (sim) => `
              <div class="card-item">
                <strong>${sim.name || ""}</strong>
                <div>${[sim.w_mm, sim.d_mm, sim.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ")}</div>
                <div>Цена/ед: ${formatNumber(sim.price_unit_ex_vat)}</div>
                <button class="ghost" data-details="${sim.id}">Подробнее</button>
              </div>
            `,
            )
            .join("")}
        </div>
      </div>
    `;
    elements.detailsContent.querySelectorAll("button[data-details]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = state.items.find((entry) => entry.id === Number(btn.dataset.details));
        if (target) showDetails(target);
      });
    });
    const compareBtn = elements.detailsContent.querySelector("#details-compare-btn");
    if (compareBtn) {
      compareBtn.addEventListener("click", () => {
        if (state.compareIds.has(item.id)) state.compareIds.delete(item.id);
        else state.compareIds.add(item.id);
        updateCompareButton();
        showDetails(item);
      });
    }
    const copyBtn = elements.detailsContent.querySelector("#details-copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => copyItemToClipboard(item));
    }
    renderPriceChart(item, similar);
    elements.detailsDrawer.classList.add("open");
  }

  function hideDetails() {
    elements.detailsDrawer.classList.remove("open");
  }

  function renderPriceChart(item, similar) {
    const canvas = elements.detailsContent.querySelector("#price-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const data = [item, ...similar.slice(0, 4)];
    const prices = data.map((entry) => entry.price_unit_ex_vat || 0);
    const maxPrice = Math.max(...prices, 1);
    const barWidth = 40;
    const gap = 16;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    data.forEach((entry, idx) => {
      const barHeight = (entry.price_unit_ex_vat || 0) / maxPrice * 120;
      const x = 20 + idx * (barWidth + gap);
      const y = 140 - barHeight;
      ctx.fillStyle = idx === 0 ? "#2f5ef6" : "#9aa6c3";
      ctx.fillRect(x, y, barWidth, barHeight);
      ctx.fillStyle = "#5f6b85";
      ctx.font = "10px sans-serif";
      ctx.fillText(`№${idx + 1}`, x + 8, 155);
    });
  }

  async function handleSearch() {
    elements.resultsLoading.classList.remove("hidden");
    const filters = getFilterValues();
    const baseItems = state.items;
    updateFilterCounts(baseItems);
    logSearchDiagnostics("filters", filters);
    let items = baseItems;
    if (filters.query) {
      const ids = state.index ? state.index.search(filters.query.toLowerCase(), { limit: 5000 }) : [];
      const idSet = new Set(ids);
      items = items.filter((item) => idSet.has(item.id));
    }
    logSearchDiagnostics("after-text-search", { count: items.length, total: baseItems.length });
    const queryItems = items;
    items = applyFilters(items, filters);
    logSearchDiagnostics("after-filters", { count: items.length });
    items = sortResults(items);
    renderResults(items);
    updateEmptyState({ queryItems, filters, results: items });
    updateActiveFilters(filters);
    saveFiltersToLocalStorage(filters);
    elements.resultsLoading.classList.add("hidden");
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
      wrapper.dataset.sheet = name;
      const status = state.previewMeta[name]?.status;
      const statusIcon = status === "ok" ? "✅" : "❌";
      const statusTitle = status === "ok" ? "Структура распознана" : "Проверьте структуру листа";
      wrapper.innerHTML = `
        <input type="checkbox" value="${name}" checked />
        <span class="sheet-status ${status}" title="${statusTitle}">${statusIcon}</span>
        <span>${name}</span>
      `;
      elements.sheetList.appendChild(wrapper);
    });
    elements.sheetOptions.classList.remove("hidden");
    elements.importBtn.disabled = false;
  }

  function updateSheetStatusBadge(sheetName, status = "ok") {
    const item = elements.sheetList.querySelector(`[data-sheet="${sheetName}"]`);
    if (!item) return;
    const badge = item.querySelector(".sheet-status");
    if (!badge) return;
    badge.classList.remove("ok", "warn", "error");
    badge.classList.add(status);
    badge.textContent = status === "ok" ? "✅" : "❌";
    badge.title = status === "ok" ? "Структура распознана" : "Проверьте структуру листа";
  }

  function classifyHeader(header) {
    if (!header) return null;
    const normalized = normalizeText(String(header));
    if (/(наимен|назван|позици|item|product)/.test(normalized)) return "name";
    if (/(цен|price|стоим)/.test(normalized)) return "price";
    if (/(размер|width|height|depth|шир|выс|глуб|длина|w|h|d)/.test(normalized)) return "dims";
    return null;
  }

  function detectPreviewHeader(rows) {
    let best = { index: 0, score: 0, classified: [] };
    rows.forEach((row, idx) => {
      const classified = (row || []).map((value) => classifyHeader(value));
      const score = classified.filter(Boolean).length;
      if (score > best.score) best = { index: idx, score, classified };
    });
    return best;
  }

  function analyzeWorkbookPreview(workbook, sheetNames) {
    const meta = {};
    sheetNames.forEach((name) => {
      const sheet = workbook.Sheets[name];
      if (!sheet) return;
      const rows = XLSX.utils.sheet_to_json(sheet, { header: 1, blankrows: false, raw: true }).slice(0, 12);
      if (!rows.length) {
        meta[name] = { status: "error", headerRowIndex: null, headers: [], classified: [] };
        return;
      }
      const detected = detectPreviewHeader(rows);
      const headerRowIndex = detected.score >= 2 ? detected.index : 0;
      const headers = rows[headerRowIndex] || [];
      const classified = detected.score >= 2 ? detected.classified : headers.map((value) => classifyHeader(value));
      const status = detected.score >= 2 ? "ok" : "warn";
      meta[name] = { status, headerRowIndex, headers, classified };
    });
    return meta;
  }

  function renderSheetPreview(workbook, sheetNames) {
    elements.sheetPreviewTabs.innerHTML = "";
    elements.sheetPreviewContent.innerHTML = "";
    if (!sheetNames.length) return;
    elements.sheetPreview.classList.remove("hidden");

    const createPreview = (name, isActive) => {
      const tab = document.createElement("button");
      tab.className = `ghost ${isActive ? "active" : ""}`;
      tab.textContent = name;
      tab.addEventListener("click", () => {
        elements.sheetPreviewTabs.querySelectorAll("button").forEach((btn) => btn.classList.remove("active"));
        tab.classList.add("active");
        renderPreviewTable(name);
      });
      elements.sheetPreviewTabs.appendChild(tab);
    };

    const renderPreviewTable = (name) => {
      const sheet = workbook.Sheets[name];
      const rows = XLSX.utils.sheet_to_json(sheet, { header: 1, blankrows: false }).slice(0, 12);
      if (!rows.length) {
        elements.sheetPreviewContent.innerHTML = "<p>Нет данных для предпросмотра.</p>";
        return;
      }
      const meta = state.previewMeta[name] || detectPreviewHeader(rows);
      const headerRowIndex = meta.headerRowIndex ?? 0;
      const headers = rows[headerRowIndex] || [];
      const classified = meta.classified || headers.map((value) => classifyHeader(value));
      const table = document.createElement("table");
      table.className = "preview-table";
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      headers.forEach((header, idx) => {
        const th = document.createElement("th");
        const type = classified[idx] || classifyHeader(header);
        if (type === "name") th.classList.add("col-name");
        if (type === "price") th.classList.add("col-price");
        if (type === "dims") th.classList.add("col-dims");
        th.textContent = header || "—";
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      rows.slice(headerRowIndex + 1).forEach((row) => {
        const tr = document.createElement("tr");
        headers.forEach((_, idx) => {
          const td = document.createElement("td");
          const type = classified[idx];
          if (type === "name") td.classList.add("col-name");
          if (type === "price") td.classList.add("col-price");
          if (type === "dims") td.classList.add("col-dims");
          td.textContent = row[idx] ?? "";
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      elements.sheetPreviewContent.innerHTML = "";
      elements.sheetPreviewContent.appendChild(table);
    };

    sheetNames.forEach((name, idx) => createPreview(name, idx === 0));
    renderPreviewTable(sheetNames[0]);
  }

  async function importWorkbook(file, sheetNames) {
    resetProgress();
    elements.progressContainer.classList.remove("hidden");
    console.info("[import] start", { file: file.name, sheets: sheetNames.length });
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
        console.info("[import] progress", payload);
        updateProgressUI(payload);
      }
      if (type === "done") {
        console.info("[import] done", payload);
        await saveMeta("summary", payload.summary);
        await saveMeta("sheetReports", payload.sheetReports);
        await saveMeta("importedAt", new Date().toISOString());
        const rowsInserted = payload.summary?.rows_inserted ?? state.items.length;
        if (!rowsInserted) {
          alert("Не удалось импортировать строки. Проверьте структуру листов или настройте колонки вручную.");
          elements.overallProgressLabel.textContent = "0%";
          elements.progressMessage.textContent = "⚠️ Данные не найдены.";
          showScreen("upload");
          worker.terminate();
          return;
        }
        buildIndex(state.items);
        updateCategoryFilter();
        resetFiltersToDefault();
        showScreen("search");
        await handleSearch();
        triggerConfetti();
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
        customMappings: state.customMappings,
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
    const debouncedSearch = debounce(handleSearch, 500);

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

    elements.mappingBtn.addEventListener("click", () => {
      const selected = getSelectedSheets();
      if (!selected.length) {
        alert("Сначала выберите лист для настройки.");
        return;
      }
      openColumnMappingModal(selected[0]);
    });

    elements.importBtn.addEventListener("click", async () => {
      const file = elements.fileInput.files[0];
      if (!file) return;
      const sheetNames = Array.from(elements.sheetList.querySelectorAll("input[type='checkbox']")).map((checkbox) => checkbox.value);
      await clearDB();
      importWorkbook(file, sheetNames);
    });

    elements.searchBtn.addEventListener("click", handleSearch);
    elements.searchInput.addEventListener("input", debouncedSearch);
    elements.searchInput.addEventListener("keyup", (event) => {
      if (event.key === "Enter") handleSearch();
    });

    Object.values(dimInputs).forEach((input) => {
      input.addEventListener("input", debouncedSearch);
    });
    Object.values(priceInputs).forEach((input) => input.addEventListener("input", debouncedSearch));
    elements.categoryFilter.addEventListener("change", handleSearch);
    elements.flagFilters.addEventListener("change", handleSearch);
    elements.closeDrawer.addEventListener("click", hideDetails);
    elements.viewTableBtn.addEventListener("click", () => setViewMode("table"));
    elements.viewCardsBtn.addEventListener("click", () => setViewMode("cards"));
    elements.exportBtn.addEventListener("click", () => exportItemsToExcel(state.lastResults, "specassist-results.xlsx"));
    elements.increaseTolBtn.addEventListener("click", () => {
      ["wTol", "dTol", "hTol"].forEach((key) => {
        const current = parseFloat(dimInputs[key].value) || 0;
        dimInputs[key].value = current + 50;
      });
      handleSearch();
    });
    elements.removeLedBtn.addEventListener("click", () => {
      const ledSelect = elements.flagFilters.querySelector("select[data-flag='has_led']");
      if (ledSelect) ledSelect.value = "";
      handleSearch();
    });
    elements.resetFiltersBtn.addEventListener("click", () => {
      resetFiltersToDefault();
      handleSearch();
    });
    elements.compareBtn.addEventListener("click", renderCompareModal);
    elements.closeCompare.addEventListener("click", () => elements.compareModal.classList.add("hidden"));
    elements.compareModal.addEventListener("click", (event) => {
      if (event.target === elements.compareModal) elements.compareModal.classList.add("hidden");
    });
    elements.themeToggle.addEventListener("click", () => {
      document.body.classList.toggle("dark");
      const isDark = document.body.classList.contains("dark");
      elements.themeToggle.textContent = isDark ? "☀️ Светлая тема" : "🌙 Тёмная тема";
    });

    elements.closeColumnMapping.addEventListener("click", closeColumnMappingModal);
    elements.columnMappingModal.addEventListener("click", (event) => {
      if (event.target === elements.columnMappingModal) closeColumnMappingModal();
    });
    elements.mappingAutoBtn.addEventListener("click", () => {
      if (!state.activeMappingSheet) return;
      const meta = state.previewMeta[state.activeMappingSheet];
      const headers = meta?.headers || [];
      const sheet = state.workbook?.Sheets?.[state.activeMappingSheet];
      const rows = sheet ? XLSX.utils.sheet_to_json(sheet, { header: 1, blankrows: false, raw: true }) : [];
      const headerRowIndex = meta?.headerRowIndex ?? 0;
      const sampleRows = rows.slice(headerRowIndex + 1, headerRowIndex + 21);
      const suggested = suggestMapping(headers, sampleRows);
      console.info("[mapping] auto-detect", { sheet: state.activeMappingSheet, suggested });
      applyMappingToModal(suggested);
      renderMappingPreview(state.activeMappingSheet);
    });
    elements.mappingSaveBtn.addEventListener("click", () => {
      if (!state.activeMappingSheet) return;
      const mapping = collectMappingFromModal();
      state.customMappings.sheets[state.activeMappingSheet] = mapping;
      if (elements.mappingApplyAll.checked) {
        state.customMappings.global = mapping;
      }
      if (!state.previewMeta[state.activeMappingSheet]) {
        state.previewMeta[state.activeMappingSheet] = { status: "ok" };
      } else {
        state.previewMeta[state.activeMappingSheet].status = "ok";
      }
      saveCustomMappings();
      updateSheetStatusBadge(state.activeMappingSheet, "ok");
      console.info("[mapping] saved", { sheet: state.activeMappingSheet, mapping, global: elements.mappingApplyAll.checked });
      closeColumnMappingModal();
    });
    elements.mappingTemplateSelect.addEventListener("change", () => {
      const idx = Number(elements.mappingTemplateSelect.value);
      if (!state.activeMappingSheet) return;
      if (Number.isNaN(idx) || !state.mappingTemplates[idx]) return;
      applyMappingToModal(state.mappingTemplates[idx].mapping);
      renderMappingPreview(state.activeMappingSheet);
    });
    elements.mappingTemplateSave.addEventListener("click", () => {
      const mapping = collectMappingFromModal();
      const name = window.prompt("Название шаблона");
      if (!name) return;
      state.mappingTemplates.push({ name, mapping });
      saveMappingTemplates();
      refreshTemplateSelect();
      elements.mappingTemplateSelect.value = String(state.mappingTemplates.length - 1);
    });
    getModalMappingSelects().forEach((select) => {
      select.addEventListener("change", () => {
        if (!state.activeMappingSheet) return;
        renderMappingPreview(state.activeMappingSheet);
      });
    });

    elements.resetBtn.addEventListener("click", async () => {
      await clearDB();
      state.items = [];
      state.index = null;
      state.compareIds.clear();
      elements.fileInput.value = "";
      elements.sheetOptions.classList.add("hidden");
      elements.fileMeta.classList.add("hidden");
      elements.progressContainer.classList.add("hidden");
      resetProgress();
      updateCompareButton();
      showScreen("upload");
    });

    const rangePairs = [
      [dimInputs.wMinRange, dimInputs.wMaxRange, dimInputs.wMin, dimInputs.wMax],
      [dimInputs.dMinRange, dimInputs.dMaxRange, dimInputs.dMin, dimInputs.dMax],
      [dimInputs.hMinRange, dimInputs.hMaxRange, dimInputs.hMin, dimInputs.hMax],
    ];
    rangePairs.forEach(([minRange, maxRange, minInput, maxInput]) => {
      minRange.addEventListener("input", () => {
        syncRangePair(minRange, maxRange, minInput, maxInput);
        debouncedSearch();
      });
      maxRange.addEventListener("input", () => {
        syncRangePair(minRange, maxRange, minInput, maxInput);
        debouncedSearch();
      });
      minInput.addEventListener("change", () => {
        setRangeFromInput(minInput, minRange, 0);
        debouncedSearch();
      });
      maxInput.addEventListener("change", () => {
        setRangeFromInput(maxInput, maxRange, maxRange.max);
        debouncedSearch();
      });
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
    state.workbook = workbook;
    state.sheetNames = sheetNames;
    state.previewMeta = analyzeWorkbookPreview(workbook, sheetNames);
    updateFileMeta(file, sheetNames);
    renderSheetList(sheetNames);
    renderSheetPreview(workbook, sheetNames);
  }

  async function init() {
    setupFlagFilters();
    setupEventListeners();
    setupSorting();
    setViewMode(window.innerWidth < 960 ? "cards" : "table");
    state.customMappings = loadCustomMappings();
    state.mappingTemplates = loadMappingTemplates();
    refreshTemplateSelect();
    const hasCache = await initFromCache();
    if (hasCache) {
      const savedFilters = loadFiltersFromLocalStorage();
      if (savedFilters) {
        applyFiltersToUI(savedFilters);
      }
      await handleSearch();
    } else {
      showScreen("upload");
    }
  }

  init();
})();
