import streamlit as st
import base64
from pathlib import Path
from streamlit.components.v1 import html
import streamlit.components.v1 as components


# --- Configuración página ---
st.set_page_config(page_title="App job data analysis", layout="wide", page_icon=":material/work_update:" ,initial_sidebar_state="collapsed")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_sidebar_style = """
    <style>
        /* Oculta la barra lateral completa */
        [data-testid="stSidebar"] {
            display: none;
        }
        /* Ajusta el área principal para usar todo el ancho */
        [data-testid="stAppViewContainer"] {
            margin-left: 0px;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

ASSETS_DIR = Path("assets")
CSS_PATH = ASSETS_DIR / "style.css"
JS_PATH = ASSETS_DIR / "script.js"
IMG_PATH = ASSETS_DIR / "ciudad-nocturna_4096x2336_xtrafondos.com.jpg"

# --- Leer archivos ---
def read_text(p: Path):
    return p.read_text(encoding="utf-8")

def image_to_data_url(p: Path):
    b = p.read_bytes()
    mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(b).decode("ascii")

css_raw = read_text(CSS_PATH)
js_raw = read_text(JS_PATH)
img_data_url = image_to_data_url(IMG_PATH)

# --- Reemplazos en CSS / JS para usar la imagen embebida ---
# En style.css hay una línea con background-image: url("..."); -> la reemplazamos
css = css_raw.replace("%%BACKGROUND_DATA_URL%%", img_data_url)

# En script.js hay un marcador %%BACKGROUND_DATA_URL%% usado para THREE loader
js = js_raw.replace("%%BACKGROUND_DATA_URL%%", img_data_url)


# --- HTML base (sin botón START, auto-arranca) ---
html_body = """
<div class="error-message" id="errorMessage"></div>
<div class="fallback-bg" id="fallbackBg"></div>

<!-- preloader -->
<div class="preloader" id="preloader">
  <span id="counter">[000]</span>
</div>

<canvas id="canvas"></canvas>

<p class="text-element description">THIS APPLICATION COLLECTS RECORDS AND DATA FROM ABANDONED WORLDS AND LOST TECHNOLOGIES, AWAITING ANALYSIS.</p>

<nav class="text-element nav-links">
  <a  target="_self">_OCCUPATION_OVERVIEW </a>
  <a  target="_self">_SUBOCCUPATION_ANALYTICS</a>
  <a  target="_self">_UNEMPLOYMENT_STATS</a>
  <a  target="_self">_INFORMAL_LABOR_TRENDS</a>
</nav>
<p class="text-element description">
  THIS APPLICATION COLLECTS RECORDS AND DATA FROM ABANDONED WORLDS AND LOST TECHNOLOGIES, AWAITING ANALYSIS.
</p>

<audio id="hoverSound" preload="auto">
  <source src="https://assets.codepen.io/7558/preloader-2s-001.mp3" type="audio/mpeg">
</audio>

<div class="text-element footer">
  <p>Err: [404 - SIGNAL LOST]<br />
    SYSTEM TIME: CYCLE 2187.42<br />
    <span style="opacity: 0.7; font-size: 0.6rem;">PRESS 'H' TO TOGGLE ANALYSIS CONTROLS</span>
  </p>
</div>

<p class="text-element division">
  PERFORMANCE ANALYSIS: <span id="fpsCounter"></span> FPS<br>
  EMPLOYMENT DATA ANALYSIS PLATFORM: ONLINE<br>
  ACCESS LEVEL: RESTRICTED.<br>
  TRACE INITIATED: SOURCE UNKNOWN.
</p>

<p class="text-element signal">_Uplink Pending...</p>

<div class="text-element central-text">
  THE SYSTEM PRESERVES RECORDS OF EXTINCT MARKETS, WORK NETWORKS.<br>
  REACTIVATED DATA FLOWS. WORK HISTORIES RECORDED IN ALGORITHMS.
</div>

<script>
document.addEventListener("DOMContentLoaded", () => {
  const hoverSound = document.getElementById("hoverSound");
  const links = document.querySelectorAll(".nav-links a");

  // Asegurarnos de preload y playsinline
  if (hoverSound) {
    hoverSound.preload = "auto";
    hoverSound.playsInline = true;
  }

  let audioUnlocked = false;

  // Intento de desbloqueo: reproduce silencioso y resetea WebAudio si existe
  const tryUnlockAudio = () => {
    if (audioUnlocked) return Promise.resolve(true);

    // 1) Intentar reproducir el elemento <audio> en silencio/clonado
    try {
      const probe = hoverSound ? hoverSound.cloneNode(true) : null;
      if (probe) {
        probe.volume = 0;
        const p = probe.play();
        if (p && p.then) {
          return p
            .then(() => {
              audioUnlocked = true;
              document.body.classList.remove("audio-locked");
              return true;
            })
            .catch(() => {
              // falló, sigue intentando con AudioContext
              return tryResumeAudioContext();
            });
        }
      }
    } catch (e) {
      // fallback a resume audio context
    }
    return tryResumeAudioContext();
  };

  // 2) Intento con AudioContext resume (si está disponible)
  const tryResumeAudioContext = () => {
    return new Promise((resolve) => {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) {
          resolve(false);
          return;
        }
        const ctx = new AudioCtx();
        // resume() requiere gesto del usuario; algunos browsers permiten resume() tras click/touch
        ctx.resume().then(() => {
          audioUnlocked = true;
          document.body.classList.remove("audio-locked");
          resolve(true);
        }).catch(() => resolve(false));
      } catch (e) {
        resolve(false);
      }
    });
  };

  // Función que activa el hover sound (solo si está desbloqueado)
  const playHover = () => {
    if (!audioUnlocked) return;
    if (!hoverSound) return;
    try {
      const s = hoverSound.cloneNode(true);
      s.volume = 0.32;
      s.play().catch(()=>{ /* ignore */ });
    } catch (e) { /* ignore */ }
  };

  // Añadir listeners de hover (siempre), pero playHover será NO-OP hasta unlock
  links.forEach(link => {
    link.addEventListener("mouseenter", () => {
      playHover();
      // pulso visual opcional: clase que se quita pronto (sin interferir)
      link.classList.add("hovered-sfx");
      setTimeout(()=>link.classList.remove("hovered-sfx"), 220);
    });
  });

  // Intentar desbloquear en los gestures comunes (solo una vez cada uno)
  const gestureUnlock = () => {
    tryUnlockAudio().then((ok) => {
      if (ok) {
        removeGestureListeners();
        hideEnableButton();
      }
    });
  };

  const onClick = () => gestureUnlock();
  const onKey = () => gestureUnlock();
  const onTouch = () => gestureUnlock();

  document.addEventListener("click", onClick, { once: true, passive: true });
  document.addEventListener("keydown", onKey, { once: true, passive: true });
  document.addEventListener("touchstart", onTouch, { once: true, passive: true });

  // Si a los 800ms no se desbloqueó, mostrar botón para habilitar audio
  let enableBtn;
  const showEnableButton = () => {
    if (enableBtn) return;
    enableBtn = document.createElement("button");
    enableBtn.innerText = "Enable audio";
    Object.assign(enableBtn.style, {
      position: "fixed",
      right: "12px",
      top: "12px",
      zIndex: 4000,
      padding: "8px 10px",
      fontSize: "0.85rem",
      background: "rgba(255,255,255,0.06)",
      color: "#fff",
      border: "1px solid rgba(255,255,255,0.12)",
      cursor: "pointer",
      borderRadius: "6px",
      backdropFilter: "blur(4px)",
    });
    enableBtn.addEventListener("click", () => {
      tryUnlockAudio().then((ok) => {
        if (ok) {
          hideEnableButton();
        } else {
          // fallback visual si aún no se puede
          enableBtn.innerText = "Tap anywhere to enable";
          setTimeout(() => { enableBtn.innerText = "Enable audio"; }, 1800);
        }
      });
    });
    document.body.appendChild(enableBtn);
  };

  const hideEnableButton = () => {
    if (!enableBtn) return;
    enableBtn.remove();
    enableBtn = null;
  };

  const removeGestureListeners = () => {
    document.removeEventListener("click", onClick);
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("touchstart", onTouch);
  };

  // marcar body si audio bloqueado para estilos opcionales
  document.body.classList.add("audio-locked");

  // Después de 800ms, si sigue bloqueado, mostrar boton
  setTimeout(async () => {
    const ok = await tryUnlockAudio();
    if (!ok) showEnableButton();
    else {
      hideEnableButton();
      removeGestureListeners();
    }
  }, 800);
});
</script>

"""

# --- Inyectar todo en la página ---
full_html = f"""
<style>
{css}
</style>

{html_body}

<!-- audio (autoplay loop; si el navegador bloquea, requerirá interacción) -->
<audio id="backgroundMusic" autoplay loop playsinline>
  <source src="https://assets.codepen.io/7558/cinematic-02.mp3" type="audio/mpeg">
</audio>

<script>
{js}
</script>
"""

# Render con streamlit (height lo podés ajustar)
st.components.v1.html(full_html, height=700, scrolling=False)


# --- Inyectar CSS desde archivo ---
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.container(border=True):

  col1, col2, col3, col4 = st.columns([2, 2, 2, 2])  # proporciones: izquierda, centro, derecha

  with col1:
      if st.button("Ocupación", key="acceso", use_container_width=True):
          st.switch_page("pages/1_ocupacion.py")

  with col2:
      if st.button("Subocupación", key="acceso1", use_container_width=True):
          st.switch_page("pages/2_subocupacion.py")

  with col3:
      if st.button("Desocupación", key="acceso2", use_container_width=True):
          st.switch_page("pages/3_desocupacion.py") 
  with col4:
      if st.button("Informalidad Laboral", key="acceso3", use_container_width=True):
          st.switch_page("pages/4_informalidad_laboral.py")

