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
    viewMode: "table",
    compareIds: new Set(),
    cartItems: [],
    worker: null,
    searchTimer: null,
    filterTimer: null,
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
    closeCompare: document.getElementById("close-compare"),
    themeToggle: document.getElementById("theme-toggle"),
    cartBtn: document.getElementById("cart-btn"),
    cartCount: document.getElementById("cart-count"),
    cartDrawer: document.getElementById("cart-drawer"),
    closeCart: document.getElementById("close-cart"),
    cartItems: document.getElementById("cart-items"),
    cartTotalItems: document.getElementById("cart-total-items"),
    cartTotalPrice: document.getElementById("cart-total-price"),
    cartAvgPrice: document.getElementById("cart-avg-price"),
    cartTotalVolume: document.getElementById("cart-total-volume"),
    cartExportBtn: document.getElementById("cart-export-btn"),
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
    const { sheetIndex, sheetName, rowsTotal, rowsInserted, rowsSkipped, sheetsTotal } = payload;
    state.progress.sheetsTotal = sheetsTotal;
    state.progress.sheetsDone = sheetIndex + 1;
    state.progress.rowsTotal += rowsTotal;
    state.progress.rowsInserted += rowsInserted;
    state.progress.rowsSkipped += rowsSkipped;

    const progressPercent = Math.round((state.progress.sheetsDone / sheetsTotal) * 100);
    elements.overallProgress.value = progressPercent;
    elements.overallProgressLabel.textContent = `${progressPercent}%`;
    if (progressPercent < 30) elements.progressMessage.textContent = "🔍 Сканируем листы...";
    else if (progressPercent < 80) elements.progressMessage.textContent = "📊 Индексируем данные...";
    else elements.progressMessage.textContent = "✨ Готово!";

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
    elements.resultsSummary.textContent = `Найдено: ${items.length}`;
    elements.resultsEmpty.classList.toggle("hidden", items.length > 0);

    const renderTableSlice = (start, end) => {
      elements.resultsTableBody.innerHTML = "";
      const fragment = document.createDocumentFragment();
      const total = items.length;
      const topSpacer = document.createElement("tr");
      topSpacer.className = "spacer-row";
      topSpacer.innerHTML = `<td colspan="10" style="height:${start * 44}px"></td>`;
      fragment.appendChild(topSpacer);
      items.slice(start, end).forEach((item) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><input type="checkbox" data-compare="${item.id}" ${state.compareIds.has(item.id) ? "checked" : ""} /></td>
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
        tr.addEventListener("click", (event) => {
          if (event.target.matches("input[type='checkbox']")) return;
          showDetails(item);
        });
        fragment.appendChild(tr);
      });
      const bottomSpacer = document.createElement("tr");
      bottomSpacer.className = "spacer-row";
      bottomSpacer.innerHTML = `<td colspan="10" style="height:${Math.max(total - end, 0) * 44}px"></td>`;
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
          <div class="dims-icon">📦 ${[item.w_mm, item.d_mm, item.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ")}</div>
          <div>Цена/ед: <strong>${formatNumber(item.price_unit_ex_vat)}</strong></div>
          <div class="card-badges">${renderBadgeChips(item) || ""}</div>
          <div class="cards-actions">
            <button class="ghost" data-details="${item.id}">Подробнее</button>
            <button class="ghost" data-cart="${item.id}">В корзину</button>
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
      elements.cardsView.querySelectorAll("button[data-cart]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const item = items.find((entry) => entry.id === Number(btn.dataset.cart));
          if (item) addToCart(item);
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

    const renderWithVirtualization = () => {
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
    elements.tableWrap.onscroll = renderWithVirtualization;
    elements.cardsView.onscroll = renderWithVirtualization;
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
    const minPrice = prices.length ? Math.min(...prices) : null;
    const maxPrice = prices.length ? Math.max(...prices) : null;

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
          cell.textContent = formatNumber(item.price_per_m2);
        } else {
          cell.textContent = item[row.key] || "—";
        }
        rowEl.appendChild(cell);
      });
      elements.compareTable.appendChild(rowEl);
    });
    elements.compareModal.classList.remove("hidden");
  }

  function addToCart(item) {
    if (!state.cartItems.find((entry) => entry.id === item.id)) {
      state.cartItems.push(item);
      updateCartUI();
    }
    elements.cartDrawer.classList.add("open");
  }

  function removeFromCart(id) {
    state.cartItems = state.cartItems.filter((item) => item.id !== id);
    updateCartUI();
  }

  function updateCartUI() {
    elements.cartItems.innerHTML = "";
    let totalPrice = 0;
    let totalAreaPrice = 0;
    let areaCount = 0;
    let totalVolume = 0;

    state.cartItems.forEach((item) => {
      const card = document.createElement("div");
      card.className = "cart-item";
      card.innerHTML = `
        <strong>${item.name || "Без названия"}</strong>
        <span>Цена/ед: ${formatNumber(item.price_unit_ex_vat)}</span>
        <span>Размеры: ${[item.w_mm, item.d_mm, item.h_mm].map((v) => (v ? Math.round(v) : "—")).join(" × ")}</span>
        <button class="ghost" data-remove="${item.id}">Удалить</button>
      `;
      elements.cartItems.appendChild(card);
      if (Number.isFinite(item.price_unit_ex_vat)) totalPrice += item.price_unit_ex_vat;
      if (Number.isFinite(item.price_per_m2)) {
        totalAreaPrice += item.price_per_m2;
        areaCount += 1;
      }
      if (item.w_mm && item.d_mm && item.h_mm) {
        totalVolume += (item.w_mm / 1000) * (item.d_mm / 1000) * (item.h_mm / 1000);
      }
    });

    elements.cartItems.querySelectorAll("button[data-remove]").forEach((btn) => {
      btn.addEventListener("click", () => removeFromCart(Number(btn.dataset.remove)));
    });

    elements.cartCount.textContent = state.cartItems.length;
    elements.cartTotalItems.textContent = state.cartItems.length;
    elements.cartTotalPrice.textContent = formatNumber(totalPrice);
    elements.cartAvgPrice.textContent = areaCount ? formatNumber(totalAreaPrice / areaCount) : "—";
    elements.cartTotalVolume.textContent = totalVolume ? `${totalVolume.toFixed(2)} м³` : "—";
  }

  function exportItemsToExcel(items, filename) {
    const rows = items.map((item) => ({
      name: item.name,
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
    const compareBtnLabel = state.compareIds.has(item.id) ? "Убрать из сравнения" : "Добавить в сравнение";
    elements.detailsContent.innerHTML = `
      <div class="details-section">
        <h3>${item.name || ""}</h3>
        <p>${item.description || ""}</p>
        <div class="details-actions">
          <button id="details-compare-btn" class="ghost">${compareBtnLabel}</button>
          <button id="details-cart-btn" class="ghost">Добавить в корзину</button>
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
    const cartBtn = elements.detailsContent.querySelector("#details-cart-btn");
    if (cartBtn) {
      cartBtn.addEventListener("click", () => addToCart(item));
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
    let items = state.items;
    if (filters.query) {
      const ids = state.index ? state.index.search(filters.query.toLowerCase(), { limit: 5000 }) : [];
      const idSet = new Set(ids);
      items = items.filter((item) => idSet.has(item.id));
    }
    updateFilterCounts(items);
    items = applyFilters(items, filters);
    items = sortResults(items);
    renderResults(items);
    updateActiveFilters(filters);
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
      wrapper.innerHTML = `<input type="checkbox" value="${name}" checked /> ${name}`;
      elements.sheetList.appendChild(wrapper);
    });
    elements.sheetOptions.classList.remove("hidden");
    elements.importBtn.disabled = false;
  }

  function classifyHeader(header) {
    if (!header) return null;
    const normalized = normalizeText(String(header));
    if (/(наимен|назван|позици|item|product)/.test(normalized)) return "name";
    if (/(цен|price|стоим)/.test(normalized)) return "price";
    if (/(размер|width|height|depth|шир|выс|глуб|длина|w|h|d)/.test(normalized)) return "dims";
    return null;
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
      const rows = XLSX.utils.sheet_to_json(sheet, { header: 1 }).slice(0, 10);
      if (!rows.length) {
        elements.sheetPreviewContent.innerHTML = "<p>Нет данных для предпросмотра.</p>";
        return;
      }
      const headers = rows[0];
      const table = document.createElement("table");
      table.className = "preview-table";
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      headers.forEach((header) => {
        const th = document.createElement("th");
        const type = classifyHeader(header);
        if (type === "name") th.classList.add("col-name");
        if (type === "price") th.classList.add("col-price");
        if (type === "dims") th.classList.add("col-dims");
        th.textContent = header || "—";
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      rows.slice(1).forEach((row) => {
        const tr = document.createElement("tr");
        headers.forEach((_, idx) => {
          const td = document.createElement("td");
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
    const debouncedSearch = debounce(handleSearch, 300);
    const throttledSearch = throttle(handleSearch, 200);

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
    elements.searchInput.addEventListener("input", debouncedSearch);
    elements.searchInput.addEventListener("keyup", (event) => {
      if (event.key === "Enter") handleSearch();
    });

    Object.values(dimInputs).forEach((input) => {
      input.addEventListener("change", throttledSearch);
    });
    Object.values(priceInputs).forEach((input) => input.addEventListener("change", throttledSearch));
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
    elements.cartBtn.addEventListener("click", () => elements.cartDrawer.classList.add("open"));
    elements.closeCart.addEventListener("click", () => elements.cartDrawer.classList.remove("open"));
    elements.cartExportBtn.addEventListener("click", () => exportItemsToExcel(state.cartItems, "specassist-cart.xlsx"));

    elements.resetBtn.addEventListener("click", async () => {
      await clearDB();
      state.items = [];
      state.index = null;
      state.compareIds.clear();
      state.cartItems = [];
      elements.fileInput.value = "";
      elements.sheetOptions.classList.add("hidden");
      elements.fileMeta.classList.add("hidden");
      elements.progressContainer.classList.add("hidden");
      resetProgress();
      updateCartUI();
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
        throttledSearch();
      });
      maxRange.addEventListener("input", () => {
        syncRangePair(minRange, maxRange, minInput, maxInput);
        throttledSearch();
      });
      minInput.addEventListener("change", () => {
        setRangeFromInput(minInput, minRange, 0);
        throttledSearch();
      });
      maxInput.addEventListener("change", () => {
        setRangeFromInput(maxInput, maxRange, maxRange.max);
        throttledSearch();
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
    updateFileMeta(file, sheetNames);
    renderSheetList(sheetNames);
    renderSheetPreview(workbook, sheetNames);
  }

  async function init() {
    setupFlagFilters();
    setupEventListeners();
    setupSorting();
    updateCartUI();
    setViewMode(window.innerWidth < 960 ? "cards" : "table");
    const hasCache = await initFromCache();
    if (!hasCache) showScreen("upload");
  }

  init();
})();
