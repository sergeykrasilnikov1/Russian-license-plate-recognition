/*
 * Run this file in the browser console while a PlatesMania gallery page is
 * open and Cloudflare verification has already completed.
 *
 * It does not copy cookies or bypass verification. Requests are made
 * sequentially by the already-open same-origin browser session.
 */
(async () => {
  "use strict";

  const PAGE_COUNT = 10;
  const DELAY_MS = 2000;
  const CARD_SELECTOR = ".panel.panel-grey";
  const VIEWER_ID = "platesmania-combined-viewer";

  if (location.hostname !== "platesmania.com") {
    throw new Error("Откройте страницу https://platesmania.com/ru/gallery.php перед запуском");
  }

  document.getElementById(VIEWER_ID)?.remove();

  const currentUrl = new URL(location.href);
  const firstStart = Number.parseInt(currentUrl.searchParams.get("start") || "0", 10);
  if (!Number.isInteger(firstStart) || firstStart < 0) {
    throw new Error("Некорректный параметр start в адресе страницы");
  }

  const viewer = document.createElement("section");
  viewer.id = VIEWER_ID;
  Object.assign(viewer.style, {
    position: "relative",
    zIndex: "2147483647",
    background: "#f4f6f8",
    color: "#17202a",
    padding: "16px",
    borderBottom: "4px solid #2c3e50",
    fontFamily: "system-ui, sans-serif",
  });

  const controls = document.createElement("div");
  Object.assign(controls.style, {
    position: "sticky",
    top: "0",
    zIndex: "2",
    display: "flex",
    gap: "12px",
    alignItems: "center",
    padding: "12px",
    marginBottom: "16px",
    background: "#ffffff",
    boxShadow: "0 2px 8px rgba(0,0,0,.15)",
  });

  const status = document.createElement("strong");
  status.textContent = "Подготовка…";
  status.style.flex = "1";

  const downloadButton = document.createElement("button");
  downloadButton.type = "button";
  downloadButton.textContent = "Скачать HTML для парсера";
  downloadButton.disabled = true;

  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.textContent = "Закрыть";
  closeButton.addEventListener("click", () => viewer.remove());

  const grid = document.createElement("div");
  Object.assign(grid.style, {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(420px, 1fr))",
    gap: "14px",
    alignItems: "start",
  });

  controls.append(status, downloadButton, closeButton);
  viewer.append(controls, grid);
  document.body.prepend(viewer);

  const cardsHtml = [];
  const seenPhotoUrls = new Set();
  const addCards = (root, start) => {
    const cards = [...root.querySelectorAll(CARD_SELECTOR)];
    if (cards.length === 0) {
      throw new Error(
        `На странице start=${start} карточки не найдены. Возможно, Cloudflare запросил повторную проверку.`
      );
    }

    let added = 0;
    for (const card of cards) {
      const photoUrl = card.querySelector('img[src*="/m/"]')?.src;
      if (!photoUrl || seenPhotoUrls.has(photoUrl)) {
        continue;
      }
      seenPhotoUrls.add(photoUrl);
      cardsHtml.push(card.outerHTML);
      const cell = document.createElement("div");
      cell.append(card.cloneNode(true));
      grid.append(cell);
      added += 1;
    }
    if (added === 0) {
      throw new Error(
        `Страница start=${start} повторила уже загруженные карточки. Попробуйте запустить помощник заново.`
      );
    }
    return added;
  };

  try {
    let total = addCards(document, firstStart);
    status.textContent = `Загружено: 1/${PAGE_COUNT} страниц, ${total} карточек`;

    for (let offset = 1; offset < PAGE_COUNT; offset += 1) {
      await new Promise((resolve) => setTimeout(resolve, DELAY_MS));

      const start = firstStart + offset;
      const url = new URL(currentUrl);
      url.hash = "";
      url.searchParams.set("ctype", "1");
      url.searchParams.set("start", String(start));
      url.searchParams.set("_pm_batch", `${Date.now()}-${start}`);

      const response = await fetch(url, {
        cache: "no-store",
        credentials: "same-origin",
        headers: { Accept: "text/html" },
      });
      if (!response.ok) {
        throw new Error(`Страница start=${start} вернула HTTP ${response.status}`);
      }

      const html = await response.text();
      const page = new DOMParser().parseFromString(html, "text/html");
      total += addCards(page, start);
      status.textContent = `Загружено: ${offset + 1}/${PAGE_COUNT} страниц, ${total} карточек`;
    }

    downloadButton.disabled = false;
    status.textContent = `Готово: ${PAGE_COUNT} страниц, ${cardsHtml.length} карточек`;

    downloadButton.addEventListener("click", () => {
      const combinedHtml = [
        "<!doctype html>",
        '<html lang="ru"><head><meta charset="utf-8"><title>PlatesMania combined</title></head><body>',
        ...cardsHtml,
        "</body></html>",
      ].join("\n");
      const blob = new Blob([combinedHtml], { type: "text/html;charset=utf-8" });
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = `platesmania-${firstStart}-${firstStart + PAGE_COUNT - 1}.html`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    });
  } catch (error) {
    status.textContent = `Остановлено: ${error instanceof Error ? error.message : String(error)}`;
    status.style.color = "#b00020";
    console.error(error);
  }
})();
