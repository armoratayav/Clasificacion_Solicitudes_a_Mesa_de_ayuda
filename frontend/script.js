/**
 * HelpDesk AI — Lógica del frontend
 * Universidad Rafael Landívar · Inteligencia Artificial 2026
 *
 * Modo actual: backend real con Flask.
 * Asegúrate de que Flask esté corriendo en localhost:5000.
 */

const USE_BACKEND = true; // Usa el servidor Flask en http://localhost:5000

// ─── Paleta de colores por categoría ──────────────────────────────────────────
const CATEGORY_COLORS = {
  "Soporte Técnico": { bar: "#5b8af5", dim: "rgba(91,138,245,0.2)" },
  "Facturación":     { bar: "#f59e0b", dim: "rgba(245,158,11,0.2)" },
  "Consulta General":{ bar: "#22d3ee", dim: "rgba(34,211,238,0.2)" },
  "Queja":           { bar: "#f87171", dim: "rgba(248,113,113,0.2)" },
  "Cancelación":     { bar: "#a78bfa", dim: "rgba(167,139,250,0.2)" },
};

const DEFAULT_COLOR = { bar: "#8b8fa8", dim: "rgba(139,143,168,0.2)" };

// ─── Estado global ─────────────────────────────────────────────────────────────
let ticketCounter = parseInt(localStorage.getItem("hdai_counter") || "0");
let sessionTickets = 0;

// ─── Inicialización ────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  generateTicketPreview();
  setupCharCounters();
  updateStatCounter();
});

/** Genera el ID del próximo ticket y lo muestra en el header del form */
function generateTicketPreview() {
  const next = ticketCounter + 1;
  document.getElementById("ticketIdPreview").textContent =
    "TCK-" + String(next).padStart(6, "0");
}

/** Configura contadores de caracteres en tiempo real */
function setupCharCounters() {
  const subjectInput = document.getElementById("subject");
  const descInput    = document.getElementById("description");

  subjectInput.addEventListener("input", () => {
    updateCounter("subjectCount", subjectInput.value.length, 100);
  });

  descInput.addEventListener("input", () => {
    updateCounter("descCount", descInput.value.length, 1000);
  });
}

function updateCounter(elId, current, max) {
  const el = document.getElementById(elId);
  el.textContent = `${current} / ${max}`;
  el.classList.remove("warn", "over");
  if (current > max) el.classList.add("over");
  else if (current > max * 0.85) el.classList.add("warn");
}

// ─── Clasificación principal ───────────────────────────────────────────────────
async function classifyTicket() {
  const subject     = document.getElementById("subject").value.trim();
  const description = document.getElementById("description").value.trim();

  // Validación
  if (!subject || !description) {
    shake(document.querySelector(".card-form"));
    showToast("⚠️  Por favor completa el asunto y la descripción.", "warn");
    return;
  }

  if (subject.length > 100) {
    showToast("El asunto no puede superar los 100 caracteres.", "warn");
    return;
  }

  setLoading(true);

  try {
    let data;

    if (USE_BACKEND) {
      // ── Modo real: llama al servidor Flask ─────────────────────────
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject, description }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data = await response.json();

    } else {
      // ── Modo simulado ──────────────────────────────────────────────
      await sleep(1200); // Simula latencia de red

      // Genera scores aleatorios para hacer la demo más dinámica
      const categories = [
        "Soporte Técnico", "Facturación", "Consulta General",
        "Queja", "Cancelación"
      ];
      const raw = categories.map(() => Math.random());
      const total = raw.reduce((a, b) => a + b, 0);
      const normalized = raw.map(v => v / total);

      // La categoría con mayor score será la predicha
      const maxIdx = normalized.indexOf(Math.max(...normalized));

      const scores = {};
      categories.forEach((cat, i) => { scores[cat] = normalized[i]; });

      // Ticket ID real
      ticketCounter++;
      localStorage.setItem("hdai_counter", ticketCounter);

      data = {
        ticket_id: "TCK-" + String(ticketCounter).padStart(6, "0"),
        category:  categories[maxIdx],
        scores,
      };
    }

    // Mostrar resultados
    renderResult(data, subject);
    sessionTickets++;
    updateStatCounter();
    showToast("Ticket clasificado exitosamente");

  } catch (error) {
    console.error("Error al clasificar:", error);
    showToast("Error al conectar con el servidor. Verifica que Flask esté activo.", "error");
  } finally {
    setLoading(false);
  }
}

// ─── Render de resultados ──────────────────────────────────────────────────────
function renderResult(data, subject) {
  const topScore    = data.scores[data.category] ?? 0;
  const pctDisplay  = (topScore * 100).toFixed(1);
  const now         = new Date();
  const dateStr     = now.toLocaleDateString("es-GT", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit"
  });

  // Actualizar campos
  document.getElementById("resTicketId").textContent  = data.ticket_id;
  document.getElementById("resDate").textContent       = dateStr;
  document.getElementById("resCategory").textContent   = data.category;
  document.getElementById("resConfidence").textContent = `Confianza: ${pctDisplay}%`;
  document.getElementById("resSubject").textContent    = subject;

  // Actualizar preview del próximo ticket
  generateTicketPreview();

  // Renderizar barras de confianza
  renderScoreBars(data.scores, data.category);

  // Mostrar panel de resultados
  document.getElementById("resultEmpty").style.display   = "none";
  document.getElementById("resultContent").style.display = "block";

  // Scroll suave hacia el resultado en mobile
  if (window.innerWidth <= 768) {
    document.getElementById("resultCard").scrollIntoView({
      behavior: "smooth", block: "start"
    });
  }
}

/** Ordena categorías por score desc y dibuja las barras animadas */
function renderScoreBars(scores, topCategory) {
  const container = document.getElementById("scoresList");
  container.innerHTML = "";

  const sorted = Object.entries(scores)
    .sort(([, a], [, b]) => b - a);

  sorted.forEach(([category, score], idx) => {
    const isTop   = category === topCategory;
    const pct     = (score * 100).toFixed(1);
    const colors  = CATEGORY_COLORS[category] ?? DEFAULT_COLOR;

    const item = document.createElement("div");
    item.className = "score-item";
    item.innerHTML = `
      <div class="score-header">
        <span class="score-name${isTop ? " top" : ""}">${category}</span>
        <span class="score-pct${isTop ? " top" : ""}">${pct}%</span>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill"
             style="background:${colors.bar}; box-shadow:${isTop ? `0 0 8px ${colors.dim}` : 'none'}"
             data-pct="${score * 100}">
        </div>
      </div>
    `;

    container.appendChild(item);
  });

  // Animar barras después de un pequeño delay (necesario para que el DOM pinte)
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      container.querySelectorAll(".score-bar-fill").forEach(bar => {
        bar.style.width = bar.dataset.pct + "%";
      });
    });
  });
}

// ─── Reset ─────────────────────────────────────────────────────────────────────
function resetForm() {
  document.getElementById("subject").value     = "";
  document.getElementById("description").value = "";
  document.getElementById("subjectCount").textContent = "0 / 100";
  document.getElementById("descCount").textContent    = "0 / 1000";
  document.getElementById("subjectCount").className   = "char-count";
  document.getElementById("descCount").className      = "char-count";

  document.getElementById("resultContent").style.display = "none";
  document.getElementById("resultEmpty").style.display   = "flex";
  document.getElementById("scoresList").innerHTML        = "";
}

// ─── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(active) {
  const btn    = document.getElementById("btnClassify");
  const loader = document.getElementById("btnLoader");
  const text   = btn.querySelector(".btn-text");
  const icon   = btn.querySelector(".btn-icon");

  if (active) {
    btn.disabled          = true;
    text.textContent      = "Clasificando…";
    icon.style.display    = "none";
    loader.classList.add("active");
  } else {
    btn.disabled          = false;
    text.textContent      = "Clasificar Ticket";
    icon.style.display    = "flex";
    loader.classList.remove("active");
  }
}

function showToast(msg, type = "success") {
  const toast = document.getElementById("toast");
  const icon  = toast.querySelector(".toast-icon");
  const text  = document.getElementById("toastMsg");

  // Estilos por tipo
  const styles = {
    success: { icon: "✓", color: "var(--green)",  bg: "var(--green-dim)"  },
    warn:    { icon: "!", color: "var(--amber)",   bg: "var(--amber-dim)"  },
    error:   { icon: "✕", color: "var(--red)",     bg: "var(--red-dim)"    },
  };

  const s = styles[type] ?? styles.success;
  icon.textContent         = s.icon;
  icon.style.color         = s.color;
  icon.style.background    = s.bg;
  text.textContent         = msg;

  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3500);
}

function shake(el) {
  el.style.animation = "none";
  el.offsetHeight; // reflow
  el.style.animation = "shakeX 0.4s ease";
  el.addEventListener("animationend", () => {
    el.style.animation = "";
  }, { once: true });
}

function updateStatCounter() {
  document.getElementById("statTotal").textContent = sessionTickets;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── CSS: shake animation (se inyecta una sola vez) ───────────────────────────
(function injectShakeCSS() {
  const style = document.createElement("style");
  style.textContent = `
    @keyframes shakeX {
      0%,100% { transform: translateX(0); }
      20%      { transform: translateX(-6px); }
      40%      { transform: translateX(6px); }
      60%      { transform: translateX(-4px); }
      80%      { transform: translateX(4px); }
    }
  `;
  document.head.appendChild(style);
})();

// ─── Tecla Enter en el campo subject ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("subject").addEventListener("keydown", e => {
    if (e.key === "Enter") {
      e.preventDefault();
      document.getElementById("description").focus();
    }
  });

  document.getElementById("description").addEventListener("keydown", e => {
    if (e.key === "Enter" && e.ctrlKey) {
      classifyTicket();
    }
  });
});
