"""
FPS Oracle — Production-Ready Streamlit Prediction App
=======================================================
Predicts in-game FPS from CPU specs, GPU specs, and Game settings
using a pre-trained scikit-learn pipeline (.pkl).

Tech stack:
    - Streamlit  ≥ 1.35
    - SQLAlchemy ≥ 2.0 (with SQLite fallback demo data)
    - joblib     ≥ 1.3
    - pandas     ≥ 2.0

Run:
    streamlit run app.py

Environment variables (optional, override in .env or shell):
    DB_URL       SQLAlchemy connection string
                 Default: sqlite:///fps_oracle.db
    MODEL_PATH   Path to the .pkl pipeline file
                 Default: fps_pipeline.pkl
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import os
import logging
import traceback

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv

# السطر ده بيدور على ملف .env في الفولدر وبيحمل المتغيرات اللي جواه
load_dotenv() 

# دلوقتي الكود هيسحب الـ DB_URL الحقيقي من الملف، ولو ملقاهوش هيستخدم الـ SQLite كاحتياطي
DB_URL: str = os.getenv("DB_URL", "sqlite:///fps_oracle.db")
# ---------------------------------------------------------------------------
# Logging — writes to console; Streamlit Cloud surfaces this in logs tab
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page-level Streamlit config — MUST be the very first st.* call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FPS Oracle",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================================================================
# 1.  CONSTANTS & ENVIRONMENT
# ===========================================================================

# Database connection string — override via environment variable DB_URL
DB_URL: str = os.getenv("DB_URL", "sqlite:///fps_oracle.db")

# Path to the serialised sklearn Pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH: str = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "fps_pipeline.pkl"))

# Exact column order that the pipeline was trained on.
# ⚠️  Edit this list to match your actual training DataFrame columns precisely.
FEATURE_COLUMNS: list[str] = [
    'CpuName', 'CpuNumberOfCores', 'CpuNumberOfThreads', 'CpuFrequency',
    'CpuMultiplier', 'CpuMultiplierUnlocked', 'CpuProcessSize', 'CpuTDP',
    'CpuTurboClock', 'CpuCacheL1', 'CpuCacheL2', 'CpuCacheL3',
    'GpuName', 'GpuArchitecture', 'GpuBandwidth', 'GpuBaseClock',
    'GpuBoostClock', 'GpuBus.interface', 'GpuDieSize', 'GpuDirectX',
    'GpuFP32Performance', 'GpuMemoryBus', 'GpuMemorySize', 'GpuMemoryType',
    'GpuOpenCL', 'GpuOpenGL', 'GpuPixelRate', 'GpuProcessSize',
    'GpuNumberOfROPs', 'GpuShaderModel', 'GpuNumberOfShadingUnits',
    'GpuNumberOfTMUs', 'GpuTextureRate', 'GpuNumberOfTransistors',
    'GpuVulkan', 'GameName', 'GameSetting'
]
# ===========================================================================
# 2.  CUSTOM CSS — CYBERPUNK / GAMING DARK THEME
# ===========================================================================

CUSTOM_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── CSS Design Tokens ─────────────────────────────────────────── */
:root {
    --red:        #ff4b4b;
    --red-dim:    #cc2e2e;
    --red-glow:   rgba(255, 75, 75, 0.35);
    --orange:     #ff7b4b;
    --bg-void:    #080b10;
    --bg-panel:   #0d1117;
    --bg-card:    #111827;
    --bg-input:   #0f1923;
    --border:     rgba(255, 75, 75, 0.22);
    --border-dim: rgba(255, 75, 75, 0.08);
    --text-hi:    #f0f4ff;
    --text-mid:   #8fa0b8;
    --text-lo:    #4a5568;
    --font-head:  'Orbitron', monospace;
    --font-body:  'Rajdhani', sans-serif;
    --font-mono:  'Share Tech Mono', monospace;
}

/* ── Global reset ──────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-void) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-body) !important;
}

/* Animated scanline overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.07) 2px,
        rgba(0,0,0,0.07) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Header / toolbar ───────────────────────────────────────────── */
[data-testid="stHeader"] {
    background: var(--bg-void) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Main content padding ───────────────────────────────────────── */
.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1440px !important;
}

/* ── Hero title ─────────────────────────────────────────────────── */
.fps-hero {
    font-family: var(--font-head);
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 900;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-hi);
    line-height: 1.1;
}
.fps-hero span.accent {
    color: var(--red);
    text-shadow: 0 0 28px var(--red-glow), 0 0 60px rgba(255,75,75,0.15);
}
.fps-subtitle {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    letter-spacing: 0.18em;
    color: var(--text-mid);
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Divider ────────────────────────────────────────────────────── */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        var(--red) 30%,
        var(--orange) 60%,
        transparent 100%);
    margin: 1.2rem 0 1.8rem;
    box-shadow: 0 0 8px var(--red-glow);
}

/* ── Section headers ────────────────────────────────────────────── */
.section-label {
    font-family: var(--font-head);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Card panels ────────────────────────────────────────────────── */
.spec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem 1.6rem 1.8rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.spec-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,75,75,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.spec-card:hover {
    border-color: rgba(255,75,75,0.45);
    box-shadow: 0 0 24px rgba(255,75,75,0.08), inset 0 0 24px rgba(255,75,75,0.03);
}

/* Corner accent */
.spec-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 28px; height: 28px;
    border-top: 2px solid var(--red);
    border-left: 2px solid var(--red);
    border-radius: 8px 0 0 0;
}

/* ── Streamlit widget overrides ─────────────────────────────────── */
/* Labels */
.stSelectbox label,
.stNumberInput label,
.stSlider label {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-mid) !important;
    text-transform: uppercase !important;
}

/* Select boxes & number inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text-hi) !important;
    font-family: var(--font-body) !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 2px var(--red-glow) !important;
}

/* Dropdown menu */
[data-baseweb="popover"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
}
[data-baseweb="menu"] {
    background: var(--bg-card) !important;
}
[data-baseweb="option"] {
    background: var(--bg-card) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-body) !important;
}
[data-baseweb="option"]:hover {
    background: rgba(255,75,75,0.14) !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--red) !important;
    border: 2px solid var(--red) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stThumbValue"] {
    color: var(--red) !important;
    font-family: var(--font-mono) !important;
}

/* ── CTA Button ─────────────────────────────────────────────────── */
div[data-testid="stButton"] button {
    width: 100%;
    background: linear-gradient(135deg, var(--red) 0%, var(--red-dim) 100%);
    color: #fff;
    border: none;
    border-radius: 4px;
    font-family: var(--font-head) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.9rem 2rem !important;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 0 20px rgba(255,75,75,0.3), inset 0 1px 0 rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
div[data-testid="stButton"] button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%);
    pointer-events: none;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 35px rgba(255,75,75,0.5), inset 0 1px 0 rgba(255,255,255,0.12);
}
div[data-testid="stButton"] button:active {
    transform: translateY(0);
}

/* ── FPS Result Card ────────────────────────────────────────────── */
.fps-result-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--red);
    border-radius: 10px;
    padding: 2.2rem 2.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 40px rgba(255,75,75,0.18),
        0 0 80px rgba(255,75,75,0.07),
        inset 0 0 40px rgba(255,75,75,0.04);
    animation: pulse-border 2.5s ease-in-out infinite;
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 40px rgba(255,75,75,0.18), 0 0 80px rgba(255,75,75,0.07); }
    50%       { box-shadow: 0 0 55px rgba(255,75,75,0.28), 0 0 100px rgba(255,75,75,0.12); }
}
.fps-result-wrapper::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,75,75,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.fps-label {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0.3em;
    color: var(--text-mid);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.fps-number {
    font-family: var(--font-head);
    font-size: clamp(4rem, 10vw, 7rem);
    font-weight: 900;
    line-height: 1;
    color: var(--red);
    text-shadow:
        0 0 20px var(--red-glow),
        0 0 50px rgba(255,75,75,0.25),
        0 0 90px rgba(255,75,75,0.1);
    letter-spacing: 0.05em;
}
.fps-unit {
    font-family: var(--font-head);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-mid);
    letter-spacing: 0.2em;
    margin-top: 0.2rem;
}
.fps-grade {
    display: inline-block;
    margin-top: 1.1rem;
    padding: 0.35rem 1.1rem;
    border-radius: 100px;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    font-weight: 600;
    text-transform: uppercase;
}

/* ── Status / info boxes ────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.74rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}
.status-ok   { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.status-warn { background: rgba(234,179, 8,0.12); color: #facc15; border: 1px solid rgba(234,179,8,0.25); }
.status-err  { background: rgba(239, 68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--red-dim); border-radius: 3px; }

/* ── Streamlit footer / watermark ───────────────────────────────── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""

# ===========================================================================
# 3.  CACHED RESOURCE LOADERS
# ===========================================================================

@st.cache_resource(show_spinner=False)
def get_db_engine():
    """
    Create and cache a SQLAlchemy engine.

    Uses the DB_URL environment variable (defaults to a local SQLite file).
    The engine is reused across all reruns — no reconnection overhead.

    Returns
    -------
    sqlalchemy.engine.Engine | None
        Engine on success, None on failure (caller handles gracefully).
    """
    try:
        engine = create_engine(
            DB_URL,
            # For SQLite: allow shared access across Streamlit threads
            connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
            pool_pre_ping=True,   # test connection health before borrowing
            pool_recycle=3600,    # recycle connections every hour
        )
        # Lightweight connectivity probe
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database engine created successfully → %s", DB_URL)
        return engine
    except SQLAlchemyError as exc:
        logger.error("DB connection failed: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """
    Load and cache the scikit-learn Pipeline from disk using joblib.

    Parameters
    ----------
    path : str
        Filesystem path to the .pkl file.

    Returns
    -------
    sklearn.pipeline.Pipeline | None
        The fitted pipeline, or None if the file cannot be loaded.
    """
    try:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found at: {path!r}")
        pipeline = joblib.load(path)
        logger.info("Model pipeline loaded from %s", path)
        return pipeline
    except (FileNotFoundError, Exception) as exc:
        logger.error("Model loading failed: %s", exc)
        return None


@st.cache_data(show_spinner=False, ttl=600)  # refresh every 10 minutes
def fetch_dropdown_data(_engine) -> dict[str, list]:
    """
    Query the database for all distinct categorical values used in dropdowns.

    Table assumptions (adapt SQL to your actual schema):
        cpus   → columns: CpuName, CpuCores, CpuThreads, CpuBaseClock,
                           CpuBoostClock, CpuTDP
        gpus   → columns: GpuName, GpuVRAM, GpuBandwidth, GpuTDP, GpuBoostClock
        games  → columns: GameName
        game_settings → columns: SettingName   (e.g. Low, Medium, High, Ultra)

    Parameters
    ----------
    _engine : sqlalchemy.engine.Engine
        Leading underscore tells Streamlit NOT to hash this argument
        (engines are not serialisable).

    Returns
    -------
    dict[str, list]
        Keys: 'cpu_names', 'gpu_names', 'game_names', 'game_settings'
        Values: sorted lists of unique string items.
    """
    queries = {
        "cpu_names":     'SELECT DISTINCT "CpuName"   FROM cpus          ORDER BY "CpuName"',
        "gpu_names":     'SELECT DISTINCT "GpuName"   FROM gpus          ORDER BY "GpuName"',
        "game_names":    'SELECT DISTINCT "GameName"  FROM games         ORDER BY "GameName"',
        "game_settings": 'SELECT DISTINCT "SettingName" FROM game_settings ORDER BY "SettingName"',
    }
    results: dict[str, list] = {}
    try:
        with _engine.connect() as conn:
            for key, sql in queries.items():
                rows = conn.execute(text(sql)).fetchall()
                results[key] = [row[0] for row in rows]
        return results
    except SQLAlchemyError as exc:
        logger.error("Dropdown fetch failed: %s", exc)
        # Return empty lists so the UI degrades gracefully
        return {k: [] for k in queries}


@st.cache_data(show_spinner=False, ttl=600)
def fetch_cpu_specs(_engine, cpu_name: str) -> dict:
    sql = text("""
        SELECT "CpuCores", "CpuThreads", "CpuBaseClock", "CpuBoostClock", "CpuTDP"
        FROM   cpus
        WHERE  "CpuName" = :name
        LIMIT  1
    """).bindparams(name=str(cpu_name))
    try:
        with _engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        if row:
            return {
                "CpuCores":      int(row[0]) if row[0] is not None else 0,
                "CpuThreads":    int(row[1]) if row[1] is not None else 0,
                "CpuBaseClock":  float(row[2]) if row[2] is not None else 0.0,
                "CpuBoostClock": float(row[3]) if row[3] is not None else 0.0,
                "CpuTDP":        float(row[4]) if row[4] is not None else 0.0,
            }
    except Exception as exc:
        logger.error("CPU spec fetch failed: %s", exc)
    return {}

@st.cache_data(show_spinner=False, ttl=600)
def fetch_gpu_specs(_engine, gpu_name: str) -> dict:
    sql = text("""
        SELECT "GpuVRAM", "GpuBandwidth", "GpuTDP", "GpuBoostClock"
        FROM   gpus
        WHERE  "GpuName" = :name
        LIMIT  1
    """).bindparams(name=str(gpu_name))
    try:
        with _engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        if row:
            return {
                "GpuVRAM":       float(row[0]) if row[0] is not None else 0.0,
                "GpuBandwidth":  float(row[1]) if row[1] is not None else 0.0,
                "GpuTDP":        float(row[2]) if row[2] is not None else 0.0,
                "GpuBoostClock": float(row[3]) if row[3] is not None else 0.0,
            }
    except Exception as exc:
        logger.error("GPU spec fetch failed: %s", exc)
    return {}


# ===========================================================================
# 4.  DEMO DATA SEED  (SQLite fallback — remove for production)
# ===========================================================================

def seed_demo_database(engine) -> None:
    """
    Populate the SQLite demo database with sample rows so the app is
    self-contained for evaluation / development.

    Call once at startup when using the default SQLite DB_URL.
    Has no effect if tables already exist (CREATE TABLE IF NOT EXISTS).
    """
    ddl_and_data = """
    -- CPUs
    CREATE TABLE IF NOT EXISTS cpus (
        CpuName TEXT PRIMARY KEY,
        CpuCores INTEGER, CpuThreads INTEGER,
        CpuBaseClock REAL, CpuBoostClock REAL, CpuTDP REAL
    );
    INSERT OR IGNORE INTO cpus VALUES
        ('Intel Core i9-14900K',  24, 32, 3.2, 6.0, 125),
        ('Intel Core i7-13700K',  16, 24, 3.4, 5.4, 125),
        ('Intel Core i5-13600K',  14, 20, 3.5, 5.1,  95),
        ('AMD Ryzen 9 7950X',     16, 32, 4.5, 5.7, 170),
        ('AMD Ryzen 7 7700X',      8, 16, 4.5, 5.4, 105),
        ('AMD Ryzen 5 7600X',      6, 12, 4.7, 5.3, 105);

    -- GPUs
    CREATE TABLE IF NOT EXISTS gpus (
        GpuName TEXT PRIMARY KEY,
        GpuVRAM REAL, GpuBandwidth REAL, GpuTDP REAL, GpuBoostClock REAL
    );
    INSERT OR IGNORE INTO gpus VALUES
        ('NVIDIA RTX 4090',    24, 1008, 450, 2520),
        ('NVIDIA RTX 4080',    16,  717, 320, 2505),
        ('NVIDIA RTX 4070 Ti', 12,  504, 285, 2610),
        ('NVIDIA RTX 3080',    10,  760, 320, 1710),
        ('AMD RX 7900 XTX',    24,  960, 355, 2500),
        ('AMD RX 7800 XT',     16,  576, 263, 2430),
        ('AMD RX 6700 XT',     12,  384, 230, 2581);

    -- Games
    CREATE TABLE IF NOT EXISTS games (GameName TEXT PRIMARY KEY);
    INSERT OR IGNORE INTO games VALUES
        ('Cyberpunk 2077'),
        ('Call of Duty: Warzone'),
        ('Battlefield 2042'),
        ('Elden Ring'),
        ('Counter-Strike 2'),
        ('Apex Legends'),
        ('Red Dead Redemption 2'),
        ('The Witcher 3');

    -- Game settings
    CREATE TABLE IF NOT EXISTS game_settings (SettingName TEXT PRIMARY KEY);
    INSERT OR IGNORE INTO game_settings VALUES
        ('Low'), ('Medium'), ('High'), ('Ultra'), ('Ultra RT');
    """
    try:
        with engine.begin() as conn:
            for statement in ddl_and_data.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    conn.execute(text(stmt))
        logger.info("Demo database seeded successfully.")
    except SQLAlchemyError as exc:
        logger.error("Demo seed failed: %s", exc)


# ===========================================================================
# 5.  HELPER UTILITIES
# ===========================================================================

def fps_performance_grade(fps: float) -> tuple[str, str, str]:
    """
    Map a raw FPS value to a human-readable performance tier.

    Returns
    -------
    (label, css_background_colour, css_text_colour)
    """
    if fps >= 240:
        return "⚡ GODLIKE",  "rgba(255,75,75,0.18)",  "#ff4b4b"
    elif fps >= 144:
        return "🔥 ELITE",    "rgba(251,146,60,0.18)", "#fb923c"
    elif fps >= 100:
        return "✅ SMOOTH",   "rgba(34,197,94,0.18)",  "#4ade80"
    elif fps >= 60:
        return "👌 PLAYABLE", "rgba(234,179,8,0.18)",  "#facc15"
    elif fps >= 30:
        return "⚠️ CHOPPY",   "rgba(249,115,22,0.18)", "#f97316"
    else:
        return "💀 SLIDESHOW","rgba(239,68,68,0.18)",  "#f87171"


def build_feature_dataframe(
    cpu_name: str,
    cpu_specs: dict,
    gpu_name: str,
    gpu_specs: dict,
    game_name: str,
    game_setting: str,
) -> pd.DataFrame:
    """
    Assemble the single-row DataFrame that the pipeline expects.

    All column names and dtypes must match the training data exactly.

    Parameters
    ----------
    cpu_name     : str   — selected CPU model
    cpu_specs    : dict  — numeric CPU specs from DB
    gpu_name     : str   — selected GPU model
    gpu_specs    : dict  — numeric GPU specs from DB
    game_name    : str   — selected game title
    game_setting : str   — selected quality preset

    Returns
    -------
    pd.DataFrame  shape (1, len(FEATURE_COLUMNS))

    Raises
    ------
    ValueError   if any required key is missing from specs dicts.
    """
    required_cpu_keys = {"CpuCores", "CpuThreads", "CpuBaseClock", "CpuBoostClock", "CpuTDP"}
    required_gpu_keys = {"GpuVRAM", "GpuBandwidth", "GpuTDP", "GpuBoostClock"}

    missing_cpu = required_cpu_keys - set(cpu_specs.keys())
    missing_gpu = required_gpu_keys - set(gpu_specs.keys())
    if missing_cpu or missing_gpu:
        raise ValueError(
            f"Spec data incomplete — missing CPU keys: {missing_cpu}, GPU keys: {missing_gpu}"
        )

    row = {
        # ── Categorical ──────────────────────────────────────────────────
        "CpuName":      cpu_name,
        "GpuName":      gpu_name,
        "GameName":     game_name,
        "GameSetting":  game_setting,
        # ── Numeric CPU ──────────────────────────────────────────────────
        "CpuCores":      cpu_specs["CpuCores"],
        "CpuThreads":    cpu_specs["CpuThreads"],
        "CpuBaseClock":  cpu_specs["CpuBaseClock"],
        "CpuBoostClock": cpu_specs["CpuBoostClock"],
        "CpuTDP":        cpu_specs["CpuTDP"],
        # ── Numeric GPU ──────────────────────────────────────────────────
        "GpuVRAM":       gpu_specs["GpuVRAM"],
        "GpuBandwidth":  gpu_specs["GpuBandwidth"],
        "GpuTDP":        gpu_specs["GpuTDP"],
        "GpuBoostClock": gpu_specs["GpuBoostClock"],
    }

    # Enforce exact column order matching the trained pipeline
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df


# ===========================================================================
# 6.  UI COMPONENT HELPERS
# ===========================================================================

def render_section_header(icon: str, label: str) -> None:
    """Inject a styled section header above a card."""
    st.markdown(
        f'<div class="section-label">{icon}&nbsp;{label}</div>',
        unsafe_allow_html=True,
    )


def render_card_open() -> None:
    """Open a spec-card <div>."""
    st.markdown('<div class="spec-card">', unsafe_allow_html=True)


def render_card_close() -> None:
    """Close a spec-card <div>."""
    st.markdown("</div>", unsafe_allow_html=True)


def render_fps_result(fps: float) -> None:
    """Render the large stylised FPS result card."""
    grade_label, grade_bg, grade_color = fps_performance_grade(fps)
    st.markdown(
        f"""
        <div class="fps-result-wrapper">
            <div class="fps-label">◈ &nbsp; Predicted Performance &nbsp; ◈</div>
            <div class="fps-number">{fps:.0f}</div>
            <div class="fps-unit">FRAMES / SECOND</div>
            <span class="fps-grade"
                  style="background:{grade_bg}; color:{grade_color};
                         border: 1px solid {grade_color}55;">
                {grade_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# 7.  MAIN APPLICATION
# ===========================================================================

def main() -> None:
    """Entry point — renders the full Streamlit page."""

    # ── 7.1  Inject CSS ────────────────────────────────────────────────────
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── 7.2  Hero Header ───────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding: 0.6rem 0 0.2rem;">
            <div class="fps-hero">
                FPS &nbsp;<span class="accent">Oracle</span>
            </div>
            <div class="fps-subtitle">
                ◈ &nbsp; AI-Powered Frame Rate Predictor &nbsp; ◈
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── 7.3  Resource initialisation ───────────────────────────────────────
    engine = get_db_engine()
    model  = load_model(MODEL_PATH)

    # Seed demo data only when running against local SQLite (dev mode)
    if engine is not None and DB_URL.startswith("sqlite"):
        seed_demo_database(engine)

    # ── 7.4  Status badges (top-right area) ────────────────────────────────
    status_col, _ = st.columns([3, 1])
    with status_col:
        badge_html = ""

        if engine is not None:
            badge_html += '<span class="status-badge status-ok">● DB Connected</span>&nbsp;&nbsp;'
        else:
            badge_html += '<span class="status-badge status-err">✕ DB Offline</span>&nbsp;&nbsp;'

        if model is not None:
            badge_html += '<span class="status-badge status-ok">● Model Loaded</span>'
        else:
            badge_html += '<span class="status-badge status-err">✕ Model Missing</span>'

        st.markdown(badge_html, unsafe_allow_html=True)

    # ── 7.5  Early-exit guard: no DB or no model ───────────────────────────
    if engine is None:
        st.error(
            "**Database unreachable.** Check your `DB_URL` environment variable "
            "and ensure the database server is running.",
            icon="🔌",
        )
        st.stop()

    if model is None:
        st.error(
            f"**Model pipeline not found** at `{MODEL_PATH}`. "
            "Place your `.pkl` file in the app directory or set `MODEL_PATH`.",
            icon="🤖",
        )
        st.stop()

    # ── 7.6  Fetch dropdown options ────────────────────────────────────────
    with st.spinner("Loading hardware & game catalogue…"):
        dropdown_data = fetch_dropdown_data(engine)

    # Friendly fallback message if a catalogue table returned empty
    for key, items in dropdown_data.items():
        if not items:
            st.warning(
                f"⚠️ No entries found for `{key}`. "
                "Check your database tables and data.",
                icon="⚠️",
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 7.7  Three-column input layout ─────────────────────────────────────
    # دالة صغيرة لتنظيف أسماء الداتا في العرض
    def clean_name(val):
        if isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
            return val[2:-1]
        return val
    cpu_col, gpu_col, game_col = st.columns([1, 1, 1], gap="large")

    # ── 7.7.1  CPU Column ──────────────────────────────────────────────────
    with cpu_col:
        render_section_header("🖥️", "CPU Configuration")
        render_card_open()

        selected_cpu: str = st.selectbox(
            "Processor Model",
            options=dropdown_data["cpu_names"],
            index=0,
            key="cpu_select",
            format_func=clean_name,
            help="Choose the CPU installed in the target system.",
        )

        # Auto-fetch & display read-only spec preview
        cpu_specs: dict = {}
        if selected_cpu:
            cpu_specs = fetch_cpu_specs(engine, selected_cpu)

        if cpu_specs:
            spec_md = (
                f"| Spec | Value |\n"
                f"|------|-------|\n"
                f"| Cores | `{cpu_specs.get('CpuCores', '—')}` |\n"
                f"| Threads | `{cpu_specs.get('CpuThreads', '—')}` |\n"
                f"| Base Clock | `{cpu_specs.get('CpuBaseClock', '—')} GHz` |\n"
                f"| Boost Clock | `{cpu_specs.get('CpuBoostClock', '—')} GHz` |\n"
                f"| TDP | `{cpu_specs.get('CpuTDP', '—')} W` |"
            )
            st.markdown(spec_md)
        else:
            st.caption("_No spec data found for this CPU._")

        render_card_close()

    # ── 7.7.2  GPU Column ──────────────────────────────────────────────────
    with gpu_col:
        render_section_header("🎮", "GPU Configuration")
        render_card_open()

        selected_gpu: str = st.selectbox(
            "Graphics Card Model",
            options=dropdown_data["gpu_names"],
            index=0,
            key="gpu_select",
            format_func=clean_name,
            help="Choose the discrete GPU in the target system.",
        )

        # Auto-fetch & display read-only spec preview
        gpu_specs: dict = {}
        if selected_gpu:
            gpu_specs = fetch_gpu_specs(engine, selected_gpu)

        if gpu_specs:
            spec_md = (
                f"| Spec | Value |\n"
                f"|------|-------|\n"
                f"| VRAM | `{gpu_specs.get('GpuVRAM', '—')} GB` |\n"
                f"| Bandwidth | `{gpu_specs.get('GpuBandwidth', '—')} GB/s` |\n"
                f"| Boost Clock | `{gpu_specs.get('GpuBoostClock', '—')} MHz` |\n"
                f"| TDP | `{gpu_specs.get('GpuTDP', '—')} W` |"
            )
            st.markdown(spec_md)
        else:
            st.caption("_No spec data found for this GPU._")

        render_card_close()

    # ── 7.7.3  Game Column ─────────────────────────────────────────────────
    with game_col:
        render_section_header("🕹️", "Game Configuration")
        render_card_open()

        selected_game: str = st.selectbox(
            "Game Title",
            options=dropdown_data["game_names"],
            index=0,
            key="game_select",
            format_func=clean_name,
            help="Select the game to benchmark.",
        )

        selected_setting: str = st.selectbox(
            "Quality Preset",
            options=dropdown_data["game_settings"],
            index=0,
            key="setting_select",
            format_func=clean_name,
            help="The in-game graphics quality setting.",
        )

        st.markdown(
            "<br><div style='font-family:var(--font-mono);font-size:0.72rem;"
            "color:var(--text-lo);letter-spacing:0.1em;'>"
            "Resolution is handled by the pipeline pre-processor.<br>"
            "Raytracing state encoded in quality preset.</div>",
            unsafe_allow_html=True,
        )

        render_card_close()

    # ── 7.8  Predict button ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])

    with btn_col:
        predict_clicked = st.button(
            "⚡  CALCULATE FPS",
            key="predict_btn",
            use_container_width=True,
        )

    # ── 7.9  Prediction logic ──────────────────────────────────────────────
    if predict_clicked:

        # Validate that specs were actually retrieved before predicting
        if not cpu_specs:
            st.error(
                "Could not retrieve specs for the selected CPU. "
                "Ensure the `cpus` table contains a matching row.",
                icon="⚙️",
            )
            st.stop()

        if not gpu_specs:
            st.error(
                "Could not retrieve specs for the selected GPU. "
                "Ensure the `gpus` table contains a matching row.",
                icon="⚙️",
            )
            st.stop()

        try:
            # Build the feature DataFrame
            input_df = build_feature_dataframe(
                cpu_name=selected_cpu,
                cpu_specs=cpu_specs,
                gpu_name=selected_gpu,
                gpu_specs=gpu_specs,
                game_name=selected_game,
                game_setting=selected_setting,
            )

            logger.info(
                "Predicting FPS | CPU=%s | GPU=%s | Game=%s | Setting=%s",
                selected_cpu, selected_gpu, selected_game, selected_setting,
            )

            # Run pipeline inference
            prediction: float = float(model.predict(input_df)[0])
            # Clamp to a physically sensible range (never negative)
            prediction = max(0.0, prediction)

            logger.info("Predicted FPS = %.2f", prediction)

        except ValueError as exc:
            # Data assembly error (missing keys, wrong shape, etc.)
            st.error(f"**Feature construction failed:** {exc}", icon="🔧")
            logger.error("Feature error: %s\n%s", exc, traceback.format_exc())
            st.stop()

        except Exception as exc:
            # Catch-all for pipeline inference errors
            st.error(
                f"**Prediction failed:** {exc}\n\n"
                "Check that the `.pkl` pipeline was trained with the same "
                "feature columns and dtypes.",
                icon="🤖",
            )
            logger.error("Prediction error: %s\n%s", exc, traceback.format_exc())
            st.stop()

        # ── 7.10  Result display ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

        # Side-by-side: big FPS number + config summary
        result_col, summary_col = st.columns([2, 1], gap="large")

        with result_col:
            render_fps_result(prediction)

        with summary_col:
            st.markdown(
                '<div class="section-label">📋 Run Summary</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                | Parameter | Value |
                |-----------|-------|
                | **CPU** | {clean_name(selected_cpu)} |
                | **Cores / Threads** | {cpu_specs.get('CpuCores','—')} / {cpu_specs.get('CpuThreads','—')} |
                | **GPU** | {clean_name(selected_gpu)} |
                | **VRAM** | {gpu_specs.get('GpuVRAM','—')} GB |
                | **Game** | {clean_name(selected_game)} |
                | **Setting** | {clean_name(selected_setting)} |
                | **Predicted FPS** | **{prediction:.1f}** |
                """
            )

    # ── 7.11  Footer ───────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; font-family:var(--font-mono); '
        'font-size:0.7rem; color:var(--text-lo); letter-spacing:0.15em; '
        'padding: 0.6rem 0;">'
        "FPS ORACLE &nbsp;|&nbsp; POWERED BY ML &nbsp;|&nbsp; "
        "PREDICTIONS ARE ESTIMATES — ACTUAL FPS MAY VARY"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# 8.  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()