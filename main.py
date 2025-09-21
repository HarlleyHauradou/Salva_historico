import io
import os
import sys
import glob
import shutil
import sqlite3
import tempfile
import platform
from contextlib import closing
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# =========================
# Configuração da página
# =========================
st.set_page_config(page_title="ExportaHist Local", page_icon="🗂️", layout="centered")

# =========================
# Constantes / utilidades
# =========================
LOCAL_TZ = ZoneInfo("America/Sao_Paulo")
CHROME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


def chrome_time_to_datetime(chrome_time_us: int) -> datetime | None:
    if pd.isna(chrome_time_us):
        return None
    return CHROME_EPOCH + timedelta(microseconds=int(chrome_time_us))


def datetime_to_chrome_time(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("datetime_to_chrome_time: dt precisa ter timezone.")
    delta = dt.astimezone(timezone.utc) - CHROME_EPOCH
    return int(delta.total_seconds() * 1_000_000)


def os_paths_candidates() -> list[str]:
    """
    Devolve possíveis caminhos para as pastas de perfis do Chrome/Chromium por SO.
    """
    paths = []
    system = platform.system().lower()
    home = os.path.expanduser("~")

    if "windows" in system:
        local = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        base = os.path.join(local, "Google", "Chrome", "User Data")
        paths += [base]
    elif "darwin" in system:  # macOS
        base = os.path.join(home, "Library", "Application Support", "Google", "Chrome")
        paths += [base]
    else:  # Linux/Unix
        paths += [
            os.path.join(home, ".config", "google-chrome"),
            os.path.join(home, ".config", "chromium"),
        ]
    return [p for p in paths if os.path.isdir(p)]


def enumerate_profiles(base_dir: str) -> list[tuple[str, str]]:
    """
    Lista perfis (nome_apresentacao, caminho_do_arquivo_History).
    Considera 'Default', 'Profile *', 'Guest Profile', etc.
    """
    candidates = []
    # Perfis típicos
    patterns = ["Default", "Profile *", "Guest Profile", "System Profile"]
    for pat in patterns:
        for prof_dir in glob.glob(os.path.join(base_dir, pat)):
            hist_path = os.path.join(prof_dir, "History")
            if os.path.isfile(hist_path):
                name = os.path.basename(prof_dir)
                candidates.append((f"{name} ({hist_path})", hist_path))
    # Evita duplicatas preservando ordem
    seen, uniq = set(), []
    for label, path in candidates:
        if path not in seen:
            uniq.append((label, path))
            seen.add(path)
    return uniq


def read_history_from_bytes(sqlite_bytes: bytes, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """
    Lê bytes de um History.sqlite e filtra por [start_utc, end_utc] em memória.
    Usa sqlite3.deserialize() se disponível (Python ≥ 3.11); senão, NamedTemporaryFile.
    """
    start_us = datetime_to_chrome_time(start_utc)
    end_us = datetime_to_chrome_time(end_utc)

    def _query(conn: sqlite3.Connection) -> pd.DataFrame:
        q = """
            SELECT urls.url, \
                   urls.title, \
                   urls.visit_count, \
                   visits.visit_time
            FROM visits
                     JOIN urls ON urls.id = visits.url
            WHERE visits.visit_time BETWEEN ? AND ?
            ORDER BY visits.visit_time DESC
            """
        df = pd.read_sql_query(q, conn, params=(start_us, end_us))
        if not df.empty:
            df["visit_datetime_utc"] = df["visit_time"].apply(chrome_time_to_datetime)
        return df

    # Caminho A: Python 3.11+ com deserialize()
    if hasattr(sqlite3.Connection, "deserialize"):
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.deserialize("main", sqlite_bytes)
            df = _query(conn)
    else:
        # Caminho B: arquivo temporário destruído em seguida
        with tempfile.NamedTemporaryFile(prefix="hist_", suffix=".db", delete=True) as tmp:
            tmp.write(sqlite_bytes)
            tmp.flush()
            with closing(sqlite3.connect(tmp.name)) as conn:
                df = _query(conn)

    if df.empty:
        return pd.DataFrame(columns=["visit_datetime_utc", "visit_count", "title", "url"])

    # Converte para fuso local e formata
    df["visit_datetime_local"] = df["visit_datetime_utc"].dt.tz_convert(LOCAL_TZ)
    df["datetime_local"] = df["visit_datetime_local"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = df[["datetime_local", "visit_count", "title", "url"]].rename(
        columns={"visit_count": "visits"}
    )
    return out


def safe_copy_history(path: str) -> bytes:
    """
    Copia o arquivo 'History' para memória com máxima segurança:
    - Se estiver bloqueado (Chrome aberto), ainda tentamos copiar via shutil.
    - Não mantém arquivo persistente — apenas bytes em RAM.
    """
    # Dica ao usuário: o ideal é fechar o Chrome para não bloquear.
    with open(path, "rb") as f:
        return f.read()


def build_pdf(df: pd.DataFrame) -> bytes:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), rightMargin=18, leftMargin=18, topMargin=22, bottomMargin=18)
    styles = getSampleStyleSheet()

    elems = []
    elems.append(Paragraph("Chrome History Export", styles["Heading2"]))
    elems.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["Normal"]))
    elems.append(Spacer(1, 6))

    data = [["Data/Hora (local)", "Visitas", "Título", "URL"]]
    for _, r in df.iterrows():
        data.append([
            r.get("datetime_local", ""),
            str(r.get("visits", "")),
            (r.get("title") or "")[:100],
            (r.get("url") or "")[:140],
        ])

    tbl = Table(data, colWidths=[150, 60, 320, 390])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e5e5")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    elems.append(tbl)

    doc.build(elems)
    buf.seek(0)
    return buf.read()


def build_excel(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="History")
    out.seek(0)
    return out.read()


# =========================
# UI — minimalista
# =========================
st.markdown(
    """
    <h2>🗂️ ExportaHist Local</h2>
    <p style="margin-top:-8px">
      <b>Como usar (bem simples):</b><br>
      1) <b>Feche o Google Chrome</b>;<br>
      2) Escolha a(s) <b>data(s)</b> do histórico;<br>
      3) Clique em <b>Exportar</b> para gerar <b>PDF</b> ou <b>Excel</b>.
    </p>
    <small>Nada é enviado ou salvo em servidor. Tudo roda localmente nesta sessão.</small>
    """,
    unsafe_allow_html=True
)

# Descobrir perfis automaticamente
base_dirs = os_paths_candidates()
all_profiles = []
for b in base_dirs:
    all_profiles += enumerate_profiles(b)

if not all_profiles:
    st.error("Não encontrei o arquivo de histórico do Chrome neste computador.\n\n"
             "Abra o Chrome ao menos uma vez e tente novamente. Em Linux, tente também o Chromium.")
    st.stop()

profile_label = st.selectbox("Perfil do Chrome", options=[p[0] for p in all_profiles])
history_path = dict(all_profiles)[profile_label]

c1, c2 = st.columns(2)
with c1:
    date_start = st.date_input("Data inicial", value=datetime.now(LOCAL_TZ).date())
with c2:
    date_end = st.date_input("Data final", value=datetime.now(LOCAL_TZ).date())

fmt = st.radio("Formato do relatório", options=("PDF", "Excel (.xlsx)"), horizontal=True, index=0)

st.caption("O filtro considera o dia inteiro (00:00:00 a 23:59:59) no fuso America/Sao_Paulo.")

run = st.button("🚀 Exportar")

# =========================
# Execução
# =========================
if run:
    try:
        # Bordas do intervalo no fuso local, depois converter a UTC
        start_local = datetime.combine(date_start, time(0, 0, 0, tzinfo=LOCAL_TZ))
        end_local = datetime.combine(date_end, time(23, 59, 59, tzinfo=LOCAL_TZ))
        start_utc, end_utc = start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

        # Lê bytes do History local (sem upload)
        data_bytes = safe_copy_history(history_path)

        # Extrai/filtra em memória
        df = read_history_from_bytes(data_bytes, start_utc, end_utc)

        if df.empty:
            st.warning("Nenhum registro encontrado no período selecionado. "
                       "Verifique as datas ou se o histórico está habilitado no Chrome.")
        else:
            st.success(f"Gerados {len(df)} registros no intervalo selecionado.")
            if fmt.startswith("PDF"):
                payload = build_pdf(df)
                st.download_button(
                    label="⬇️ Baixar PDF",
                    data=payload,
                    file_name=f"chrome_history_{date_start}_{date_end}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                payload = build_excel(df)
                st.download_button(
                    label="⬇️ Baixar Excel (.xlsx)",
                    data=payload,
                    file_name=f"chrome_history_{date_start}_{date_end}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            with st.expander("🔎 Prévia (até 200 linhas)"):
                st.dataframe(df.head(200))

    except PermissionError:
        st.error("Permissão negada ao ler o arquivo de histórico. "
                 "Feche o Google Chrome e tente novamente.")
    except sqlite3.DatabaseError as e:
        st.error("Falha ao abrir o banco do histórico. Feche o Chrome e tente novamente.")
        st.code(str(e))
    except Exception as e:
        st.error("Ocorreu um erro inesperado.")
        st.exception(e)
