"""Web UI:  streamlit run streamlit_app.py"""
import contextlib
import io
import os
import re
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.consistency import detect_cross_paper_conflicts
from app.highlight import render_page
from app.extractor import diagnose_fields, suggest_fields_from_pages
from app.pdf_reader import read_pdf
from app import db
from app.pipeline import process_paper
from app.summary import summarize, to_number

load_dotenv(Path(__file__).resolve().parent / ".env")
try:  # on Streamlit Cloud, keys live in "secrets" instead of .env
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

st.set_page_config(page_title="TREX: Traceable Research Extractor", page_icon="🦖", layout="wide")
# The add-on that reads the browser ID (see visitor_id) is invisible: keep it running but take no space.
st.markdown("<style>.st-key-trex_visitor{position:absolute;width:0;height:0;overflow:hidden;opacity:0;"
            "pointer-events:none}</style>", unsafe_allow_html=True)

# ---------------- Public-site limits (set TREX_PUBLIC=true in the hosted app's secrets) ----------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


PUBLIC = os.getenv("TREX_PUBLIC", "").strip().lower() in ("1", "true", "yes")
LIMITS = {
    "papers": _env_int("PUBLIC_MAX_PAPERS", 2),          # papers per extraction
    "mb": _env_int("PUBLIC_MAX_MB", 10),                 # size per paper
    "pages": _env_int("PUBLIC_MAX_PAGES", 40),           # pages per paper
    "calls": _env_int("PUBLIC_MAX_CALLS", 6),            # AI requests per paper
    "session_runs": _env_int("PUBLIC_RUNS_PER_SESSION", 3),
    "daily_runs": _env_int("PUBLIC_DAILY_RUNS", 40),     # fresh extractions per day, whole site
}
DEMO_FIELDS = os.getenv("DEMO_FIELDS", "Model, Benchmark, Score")

ICON = {"verified": "✅", "mismatch": "⚠️", "unverified": "❌"}
LABEL = {"verified": "Verified", "mismatch": "Number mismatch", "unverified": "Quote not found"}

# Streamlit reruns the script on every click, so keep files and results in session state.
ss = st.session_state
ss.setdefault("tmpdir", tempfile.mkdtemp(prefix="papers_"))
ss.setdefault("rows", [])
ss.setdefault("fields", [])
ss.setdefault("summary", None)
ss.setdefault("last_run", None)  # what happened on the last Extract: errors, warnings, timing
ss.setdefault("fields_text", "")
ss.setdefault("suggestions", {})  # file name -> list of suggested field sets
ss.setdefault("diagnoses", {})    # (file name, fields) -> per-field explanation
ss.setdefault("session_runs", 0)  # fresh extractions by this visitor (public limit)
ss.setdefault("unlocked", False)  # owner entered TREX_UNLOCK: no limits for this session


def limited() -> bool:
    """Public limits apply: public site, and this session hasn't been unlocked."""
    return PUBLIC and not ss.unlocked


@st.cache_data(ttl=30, show_spinner=False)
def tiger_status():
    """Connection check for the sidebar (refreshed at most every 30 s). None if no database is set."""
    return db.check() if db.enabled() else None


def parse_fields(text: str) -> list[str]:
    """'Model, Benchmark, Score' -> ['Model', 'Benchmark', 'Score']. Ignores a typed 'ex)' prefix."""
    text = re.sub(r"^\s*(ex\)|e\.g\.|ex:)\s*", "", text or "", flags=re.I)
    return [f.strip() for f in text.split(",") if f.strip()]


def use_fields(names: list[str]) -> None:
    """Button callback: put a suggested field set into the Fields box."""
    ss.fields_text = ", ".join(names)


def get_suggestions(path: Path) -> list[dict]:
    """Ask the model which columns this paper actually reports (cached per file)."""
    if path.name not in ss.suggestions:
        try:
            ss.suggestions[path.name] = suggest_fields_from_pages(read_pdf(path))
        except Exception as e:
            st.warning(f"Couldn't get suggestions for {path.name}: {e}")
            ss.suggestions[path.name] = []
    return ss.suggestions[path.name]


STATUS_ICON = {"reported": "✅", "mentioned_only": "🟡", "not_found": "❌"}
STATUS_TEXT = {"reported": "found", "mentioned_only": "mentioned, but no values", "not_found": "not in this paper"}


def show_diagnosis(path: Path, fields: list[str]) -> None:
    """Explain field by field why no rows could be built from this paper."""
    key = (path.name, tuple(fields))
    if key not in ss.diagnoses:
        with st.spinner("Checking which fields this paper reports..."):
            try:
                ss.diagnoses[key] = diagnose_fields(read_pdf(path), fields)
            except Exception as e:
                st.caption(f"Couldn't check the fields: {e}")
                ss.diagnoses[key] = []
    diag = ss.diagnoses[key]
    if not diag:
        return
    st.markdown(f"**Why nothing came back from *{path.name}*:**")
    for d in diag:
        line = f"{STATUS_ICON[d['status']]} **{d['field']}**: {STATUS_TEXT[d['status']]}"
        if d["status"] == "reported" and d.get("example"):
            line += f', e.g. "{d["example"]}"' + (f" (p.{d['page']})" if d.get("page") else "")
        elif d.get("explanation"):
            line += f". {d['explanation']}"
        if d["status"] != "reported":
            if d.get("mentioned_on"):
                line += f" The words appear on p.{', '.join(map(str, d['mentioned_on']))}, but not as values."
            if d.get("closest"):
                line += f" Closest available: *{d['closest']}*."
        st.markdown(line)
    missing = [d["field"] for d in diag if d["status"] != "reported"]
    kept = [f for f in fields if f not in missing]
    if missing and len(kept) >= 2:
        st.markdown("Every row needs **all** the fields together, so one missing field means no rows.")
        st.button(f"Use without {', '.join(missing)}  →  {', '.join(kept)}", key=f"drop_{path.name}",
                  on_click=use_fields, args=(kept,), width="stretch")
        for d in diag:
            if d["status"] != "reported" and d.get("closest"):
                swapped = [d["closest"] if f == d["field"] else f for f in fields]
                st.button(f"Swap {d['field']} for {d['closest']}  →  {', '.join(swapped)}",
                          key=f"swap_{path.name}_{d['field']}", on_click=use_fields, args=(swapped,),
                          width="stretch")


def show_suggestions(path: Path, where: str) -> None:
    """One button per suggested column set, with what it compares and an example row."""
    sets = get_suggestions(path)
    if not sets:
        st.caption("No suggestions found for this paper.")
        return
    for i, s in enumerate(sets):
        st.button(f"Use: {', '.join(s['fields'])}", key=f"use_{where}_{path.name}_{i}",
                  on_click=use_fields, args=(s["fields"],), width="stretch")
        detail = s.get("why") or ""
        if s.get("example"):
            detail += ("  \n" if detail else "") + "e.g. " + " · ".join(f"{k}: {v}" for k, v in s["example"].items())
        if detail:
            st.caption(detail)


def is_owner() -> bool:
    """Owner view: running locally, or the public site unlocked with TREX_UNLOCK.
    The owner sees every visitor's runs and the demo; strangers see only their own History."""
    return (not PUBLIC) or ss.unlocked


# A random ID kept in the browser's localStorage, so each visitor gets their own History without accounts.
VISITOR_JS = ("(() => { try { let v = localStorage.getItem('trex_visitor');"
              " if (!v) { v = crypto.randomUUID(); localStorage.setItem('trex_visitor', v); }"
              " return v; } catch (e) { return ''; } })()")


def visitor_id() -> str:
    """This browser's ID. Until the browser answers (or if its storage is blocked), a temporary
    per-visit ID is used; runs made under it are handed over once the real ID arrives."""
    import uuid
    if "visitor" not in ss:
        ss.visitor, ss.visitor_checked = str(uuid.uuid4()), False
    if not ss.visitor_checked and db.enabled():
        try:
            from streamlit_js_eval import streamlit_js_eval
            got = streamlit_js_eval(js_expressions=VISITOR_JS, key="trex_visitor")
        except Exception:
            got = ""  # add-on missing or broken: History lasts for this visit only
        if got is not None:  # None = the browser hasn't answered yet
            ss.visitor_checked = True
            if re.fullmatch(r"[0-9a-f-]{36}", str(got)) and got != ss.visitor:
                db.move_visitor(ss.visitor, got)
                ss.visitor = got
    return ss.visitor


def open_run(run_id: int, run_fields: list[str]) -> None:
    """Reopen one saved run (History callback). Logged as a 'reopen' so it moves to the top."""
    ss.rows = db.load_runs([run_id], pdf_dir=Path(ss.tmpdir) / "library")
    ss.fields, ss.summary, ss.last_run = run_fields, None, None
    ss.open_run_ids = [run_id]
    db.log_event("reopen", [], run_id=run_id, fields=run_fields)
    db.touch_run(ss.get("visitor"), run_id)


def history_bucket(ts) -> str:
    import datetime as _dt
    days = (_dt.datetime.now(_dt.timezone.utc).date() - ts.astimezone(_dt.timezone.utc).date()).days
    return "Today" if days <= 0 else "Yesterday" if days == 1 else "Previous 7 days" if days <= 7 else "Older"


def render_history() -> tuple[list[dict], bool, list[int]]:
    """Claude-style history in the sidebar, newest first, grouped by date. Visitors see the runs
    made in their own browser; the owner sees every run.
    Returns (runs, load-several clicked, picked run ids) for the multi-open option."""
    from app.summary import canon
    st.markdown("#### 🕘 History" + (" · all visitors" if PUBLIC and is_owner() else ""))
    me = visitor_id()
    runs = db.history() if is_owner() else db.history(visitor=me)
    if not runs:
        st.caption("Your extractions will appear here. History is kept in this browser"
                   + (", so another browser or a private window starts empty." if PUBLIC else "."))
        return [], False, []
    query = st.text_input("Search history", placeholder="Search papers or fields", label_visibility="collapsed")
    shown = [r for r in runs if not query or canon(query) in canon(f"{r['title']} {r['file_name']} {' '.join(r['fields'])}")]
    limit = len(shown) if ss.get("history_all") else 12
    current = set(ss.get("open_run_ids") or [])
    last_bucket = None
    for r in shown[:limit]:
        bucket = history_bucket(r["last_used"])
        if bucket != last_bucket:
            st.caption(bucket)
            last_bucket = bucket
        mark = "▸ " if r["run_id"] in current else ""
        st.button(f"{mark}{r['title'] or r['file_name']} · {', '.join(r['fields'])}",
                  key=f"hist_{r['run_id']}", on_click=open_run, args=(r["run_id"], r["fields"]),
                  help=f"{r['rows']} results, {r['verified']} ✅ verified · last used {ago(r['last_used'])}",
                  width="stretch")
    if len(shown) > limit:
        st.button(f"Show all {len(shown)}", key="hist_more", on_click=lambda: ss.update(history_all=True))
    if query and not shown:
        st.caption("No matches.")
    picked, load = [], False
    with st.expander("Open several at once"):
        label = {r["run_id"]: f"{r['title'] or r['file_name']} · {', '.join(r['fields'])}" for r in runs}
        picked = st.multiselect("Pick runs", list(label), format_func=label.get,
                                label_visibility="collapsed", placeholder="Pick runs to compare")
        load = st.button("Open selected", disabled=not picked, width="stretch")
    return runs, load, picked


def ago(ts) -> str:
    import datetime as _dt
    secs = (_dt.datetime.now(_dt.timezone.utc) - ts).total_seconds()
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= n:
            return f"{int(secs // n)}{unit} ago"
    return "just now"


def load_demo() -> None:
    """Load every saved run with the demo fields from Tiger Cloud (button callback)."""
    key = db.fields_key([f.strip() for f in DEMO_FIELDS.split(",") if f.strip()])
    runs = [r for r in db.list_runs() if db.fields_key(r["fields"]) == key]
    rows = db.load_runs([r["run_id"] for r in runs], pdf_dir=Path(ss.tmpdir) / "library") if runs else []
    ss.rows, ss.summary, ss.last_run = rows, None, None
    ss.fields = runs[0]["fields"] if runs else []
    ss.demo_loaded = len(runs)


def public_block_reason(uploads) -> str | None:
    """Why a live extraction isn't allowed on the public site right now (None = allowed)."""
    if not limited():
        return None
    if len(uploads) > LIMITS["papers"]:
        return f"The public demo reads up to {LIMITS['papers']} papers at a time. Remove some and try again."
    if ss.session_runs >= LIMITS["session_runs"]:
        return (f"You've used this session's {LIMITS['session_runs']} live extractions. "
                "Explore the demo results, or run TREX locally for unlimited use.")
    recent = db.runs_since(24)
    if recent is not None and recent >= LIMITS["daily_runs"]:
        return "Today's live-extraction limit for the public site has been reached. Try the demo results instead."
    return None


def public_file_problem(up) -> str | None:
    """Size/page checks for one uploaded paper on the public site."""
    if not limited():
        return None
    if up.size > LIMITS["mb"] * 1024 * 1024:
        return f"{up.name} is over {LIMITS['mb']} MB (public demo limit)"
    try:
        import pymupdf
        with pymupdf.open(stream=up.getvalue(), filetype="pdf") as doc:
            if doc.page_count > LIMITS["pages"]:
                return f"{up.name} has {doc.page_count} pages; the public demo reads up to {LIMITS['pages']}"
    except Exception:
        return f"{up.name} couldn't be opened as a PDF"
    return None


# ---------------- Sidebar: input ----------------
with st.sidebar:
    st.header("1. Papers")
    uploads = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    if limited():
        st.caption(f"Public demo: up to {LIMITS['papers']} papers per run, {LIMITS['pages']} pages / "
                   f"{LIMITS['mb']} MB each, {LIMITS['session_runs']} live runs per visit.")
    elif PUBLIC:
        st.caption("🔓 Unlocked: no limits for this session.")
    st.header("2. Fields")
    fields_text = st.text_input("Comma-separated", key="fields_text",
                                placeholder="ex) Model, Benchmark, Score",
                                help="Any columns you want, e.g. 'Drug, Dose, Outcome' or 'Method, Sample size, p-value'. "
                                     "Put the main number last.")
    if st.button("✨ Suggest fields from my paper", disabled=not uploads, width="stretch",
                 help="Reads the first uploaded paper and proposes columns it actually reports"):
        first = Path(ss.tmpdir) / uploads[0].name
        first.write_bytes(uploads[0].getvalue())
        with st.spinner("Reading the paper..."):
            get_suggestions(first)
        ss.suggest_for = first.name
    if uploads and ss.get("suggest_for") == uploads[0].name:
        st.caption("Fields this paper reports (click to use):")
        show_suggestions(Path(ss.tmpdir) / uploads[0].name, "sidebar")
    if uploads and not parse_fields(fields_text):
        st.caption("Type the columns you want, or use ✨ Suggest fields.")
    use_cache = True
    if db.enabled():
        use_cache = st.checkbox("Reuse saved results", value=True,
                                help="If a paper was already extracted with the same fields, load it from "
                                     "Tiger Cloud instantly instead of calling the AI again.")
    run = st.button("Extract", type="primary", disabled=not (uploads and parse_fields(fields_text)),
                    width="stretch")

    st.divider()
    lib_runs, load_lib, picked_runs = [], False, []
    if db.enabled():
        lib_runs, load_lib, picked_runs = render_history()
        st.divider()

    saved, pdf_dir, load = None, "papers", False
    if not PUBLIC:  # reads files from the server's disk, so it's local-only
        with st.expander("Or load a results CSV"):
            st.caption("Open a CSV from `python cli.py ...` instantly, without re-running the AI. "
                       "PDF pages are looked up in the folder below.")
            saved = st.file_uploader("Results CSV", type="csv", label_visibility="collapsed")
            pdf_dir = st.text_input("PDF folder", "papers")
            load = st.button("Load results", disabled=not saved, width="stretch")

    unlock_code = os.getenv("TREX_UNLOCK", "").strip()
    if PUBLIC and unlock_code and not ss.unlocked:
        with st.expander("🔑 Owner unlock"):
            code = st.text_input("Unlock code", type="password", label_visibility="collapsed",
                                 placeholder="Unlock code")
            if code:
                import hmac
                if hmac.compare_digest(code.strip(), unlock_code):
                    ss.unlocked = True
                    st.rerun()
                else:
                    st.caption("Wrong code.")

    status = tiger_status()
    if status is not None:
        st.caption(f"🐯 Tiger Cloud: connected · {status['papers']} papers, {status['rows']} results saved"
                   if status["ok"] else f"🐯 Tiger Cloud: not connected ({status['error'][:80]})")

META_COLS = {"paper", "file", "quote", "page", "found_page", "match_score", "status",
             "inconsistency", "alignment", "path", "setting", "cached"}

if load_lib and picked_runs:
    with st.spinner("Loading from Tiger Cloud..."):
        lib_rows = db.load_runs(picked_runs, pdf_dir=Path(ss.tmpdir) / "library")
    picked = [r for r in lib_runs if r["run_id"] in picked_runs]
    lib_fields = []
    for r in picked:  # union of the runs' fields, in order
        lib_fields += [f for f in r["fields"] if f not in lib_fields]
    ss.rows, ss.fields, ss.summary, ss.last_run = lib_rows, lib_fields, None, None
    ss.open_run_ids = list(picked_runs)
    for rid in picked_runs:
        db.log_event("reopen", [], run_id=rid)
        db.touch_run(ss.get("visitor"), rid)
    if not lib_rows:
        st.warning("Those runs have no saved results.")

if load and saved:
    loaded = pd.read_csv(saved, dtype=str, keep_default_na=False)
    for col in ("found_page", "match_score"):
        if col in loaded:
            loaded[col] = pd.to_numeric(loaded[col], errors="coerce")
    if "inconsistency" in loaded:
        loaded["inconsistency"] = loaded["inconsistency"].replace("", None)
    loaded_fields = [c for c in loaded.columns if c not in META_COLS]
    loaded["path"] = loaded["file"].map(lambda f: str(Path(pdf_dir) / f)) if "file" in loaded else ""
    missing = sorted({p for p in loaded["path"] if p and not Path(p).exists()})
    ss.rows, ss.fields, ss.summary = loaded.replace("", None).to_dict("records"), loaded_fields, None
    ss.last_run = None
    if missing:
        st.warning(f"PDFs not found (source pages won't render): {', '.join(missing[:5])}")

st.title("🦖 TREX")
st.markdown("**Traceable Research Extractor**: pull the numbers you need out of research papers, "
            "with every value traced to its exact page and quote.")

blocked = public_block_reason(uploads) if run else None
if blocked:
    st.warning(blocked)
    run = False

if run:
    fields = parse_fields(fields_text)
    rows, failed, per_paper, cached_papers = [], [], {}, []
    max_calls = LIMITS["calls"] if limited() else None
    log = io.StringIO()  # collects warnings (rate limits, cut-off replies) that normally go to the terminal
    start = time.time()
    bar = st.progress(0.0, text="Starting...")
    with contextlib.redirect_stderr(log):
        for i, up in enumerate(uploads):
            bar.progress(i / len(uploads),
                         text=f"Reading {up.name} ({i + 1}/{len(uploads)}) · starting...")

            def show(done, total, pages_txt, i=i, name=up.name):
                bar.progress(min((i + done / total) / len(uploads), 1.0),
                             text=f"Reading {name} ({i + 1}/{len(uploads)}) · {done}/{total} parts done "
                                  f"(last: {pages_txt})")
            problem = public_file_problem(up)
            if problem:
                failed.append(problem)
                continue
            path = Path(ss.tmpdir) / up.name
            path.write_bytes(up.getvalue())
            try:
                got = process_paper(path, fields, use_cache=use_cache, on_progress=show, max_calls=max_calls,
                                    visitor=ss.get("visitor"))
                for r in got:
                    r["path"] = str(path)
                rows.extend(got)
                per_paper[up.name] = len(got)
                if got and got[0].get("cached"):
                    cached_papers.append(up.name)
                else:
                    ss.session_runs += 1
            except Exception as e:
                failed.append(f"{up.name}: {type(e).__name__}: {e}")
    bar.empty()
    tiger_status.clear()
    ss.rows, ss.fields, ss.summary = rows, fields, None
    ss.last_run = {"failed": failed, "warnings": [w.strip() for w in log.getvalue().splitlines() if w.strip()],
                   "per_paper": per_paper, "seconds": time.time() - start, "fields": fields,
                   "cached": cached_papers}
    ss.open_run_ids = []
    if db.enabled():
        st.rerun()  # redraw so the sidebar History includes this run (the report above is kept in session)

# Always report what the last run did, so an empty result never looks like "nothing happened"
lr = ss.last_run
if lr:
    for f in lr["failed"]:
        st.error(f"Failed: {f}")
    if lr["warnings"]:
        with st.expander(f"⚠️ {len(lr['warnings'])} warning(s) during extraction", expanded=not ss.rows):
            for w in lr["warnings"]:
                st.write(w)
    if not ss.rows:
        empty = [name for name, n in lr["per_paper"].items() if n == 0]
        if lr["failed"]:
            st.error("Extraction failed, so no results came back. See the error above.")
            st.markdown(
                "- Check `LLM_PROVIDER`, `LLM_MODEL` and the API key in `.env` (or in the app's secrets)\n"
                "- If it mentions a rate or token limit, wait a minute and try again")
        else:
            asked = ", ".join(lr.get("fields") or [])
            st.warning(f"**Nothing matched your fields** ({asked}). TREX won't invent values.")
            if empty:
                show_diagnosis(Path(ss.tmpdir) / empty[0], lr.get("fields") or [])
                st.markdown(f"**Or pick fields that *{empty[0]}* does report.** "
                            "Click one, then press **Extract** again:")
                show_suggestions(Path(ss.tmpdir) / empty[0], "empty")
            with st.expander("Other possible causes"):
                st.markdown(
                    "- A rate or token limit was hit: see the warnings above, or wait a minute and retry\n"
                    "- Scanned PDF (images only): text can't be read yet")
        st.stop()
    else:
        if lr["per_paper"] and any(n == 0 for n in lr["per_paper"].values()):
            missing = [name for name, n in lr["per_paper"].items() if n == 0]
            st.info(f"No matches for your fields in: {', '.join(missing)}. "
                    "Use **✨ Suggest fields** with that paper first to see what it reports.")
        cached_note = (f" · {len(lr['cached'])} of {len(lr['per_paper'])} papers loaded from Tiger Cloud "
                       f"(no AI call)") if lr.get("cached") else ""
        st.caption(f"Last extraction: {len(ss.rows)} results in {lr['seconds']:.0f}s{cached_note}")

if not ss.rows:
    st.info("**How it works:** 1) upload papers, 2) type the columns you want, 3) press **Extract**. "
            "Every value comes with a quote from the paper, and code (not AI) checks the quote is really there.")
    if db.enabled() and is_owner():
        st.button(f"▶ Try the demo: saved results for {DEMO_FIELDS}, loads instantly",
                  type="primary", on_click=load_demo)
        if ss.get("demo_loaded") == 0:
            st.caption("No demo results saved yet. Run the papers once with those fields to create them.")
    st.stop()

fields = ss.fields
df = pd.DataFrame(ss.rows)
for f in fields:
    if f not in df:
        df[f] = None

# ---------------- Metrics ----------------
counts = df["status"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Results", len(df))
c2.metric("✅ Verified", int(counts.get("verified", 0)))
c3.metric("⚠️ Mismatch", int(counts.get("mismatch", 0)))
c4.metric("❌ Not found", int(counts.get("unverified", 0)))

if "inconsistency" not in df:
    df["inconsistency"] = None
conflicts = detect_cross_paper_conflicts(df[df["status"] == "verified"].to_dict("records"), fields)
n_intra = int(df["inconsistency"].notna().sum())

tab_table, tab_conflicts, tab_board, tab_summary = st.tabs(
    ["Table", f"Conflicts ({len(conflicts) + (n_intra > 0)})", "Leaderboard", "Summary"])

# ---------------- Table tab ----------------
with tab_table:
    f1, f2, f3 = st.columns([2, 2, 3])
    status_pick = f1.multiselect("Status", list(LABEL), default=list(LABEL), format_func=lambda s: f"{ICON[s]} {LABEL[s]}")
    paper_pick = f2.multiselect("Paper", sorted(df["paper"].unique()))
    search = f3.text_input("Search values", placeholder="e.g. ResNet")

    view = df[df["status"].isin(status_pick)]
    if paper_pick:
        view = view[view["paper"].isin(paper_pick)]
    if search:
        mask = view[fields].astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view = view[mask]

    has_setting = "setting" in view and view["setting"].notna().any()
    show = view[["status", "paper", *(["setting"] if has_setting else []), *fields,
                 "found_page", "inconsistency"]].copy()
    show.insert(0, " ", show.pop("status").map(ICON))
    show["inconsistency"] = show["inconsistency"].map(lambda v: "🔀 differs in paper" if isinstance(v, str) else "")
    show = show.rename(columns={"found_page": "page", "inconsistency": "flag", "setting": "table / setting"})

    st.caption("Click a row to see its source.")
    event = st.dataframe(show, width="stretch", hide_index=True,
                         on_select="rerun", selection_mode="single-row")
    st.download_button("Download CSV", view.drop(columns=["path"], errors="ignore").to_csv(index=False),
                       "results.csv", "text/csv")

    picked = event.selection.rows if event and event.selection else []
    if picked:
        row = view.iloc[picked[0]]
        st.divider()
        left, right = st.columns([2, 3])
        with left:
            st.subheader(f"{ICON[row['status']]} {LABEL[row['status']]}")
            st.markdown(f"**Paper:** {row['paper']}")
            if isinstance(row.get("setting"), str) and row.get("setting"):
                st.markdown(f"**Table / setting:** {row['setting']}")
            for f in fields:
                st.markdown(f"**{f}:** {row[f] if pd.notna(row[f]) else '—'}")
            st.markdown("**Evidence quote:**")
            st.markdown(f"> {row.get('quote') or '—'}")
            st.caption(f"Page {row['found_page']} · match score {row['match_score']}")
            if isinstance(row.get("inconsistency"), str):
                st.info("🔀 This paper reports a different number for the same result elsewhere. "
                        "See the Conflicts tab.")
            if row["status"] == "mismatch":
                st.warning("The quote exists, but a number in the values isn't in it. Check by hand.")
            elif row["status"] == "unverified":
                st.error("This quote wasn't found in the paper. The AI may have made it up.")
        with right:
            page_no = int(float(row["found_page"])) if pd.notna(row["found_page"]) else int(float(row.get("page") or 1))
            try:
                png, hit = render_page(row["path"], page_no, row.get("quote") or "")
                st.image(png, caption=f"Page {page_no}" + ("" if hit else " (quote not located on page)"),
                         width="stretch")
            except Exception as e:
                st.info(f"Couldn't render page: {e}")

# ---------------- Conflicts tab ----------------
with tab_conflicts:
    st.caption("Same result, different numbers. Rows from tables with different settings (e.g. another "
               "test problem or true parameter) are not compared. Treat these as **worth a look**, not errors.")
    intra = df[df["inconsistency"].notna()]
    st.subheader(f"Within one paper ({len(intra)} rows)")
    if intra.empty:
        st.write("None found.")
    else:
        intra_cols = ["paper", *(["setting"] if "setting" in intra and intra["setting"].notna().any() else []),
                      *fields, "found_page", "quote"]
        st.dataframe(intra[intra_cols].rename(columns={"found_page": "page", "setting": "table / setting"}),
                     hide_index=True, width="stretch")
    st.subheader(f"Across papers ({len(conflicts)})")
    st.caption("Uses only ✅ verified rows.")
    if not conflicts:
        st.write("None found.")
    for c in conflicts:
        with st.container(border=True):
            st.markdown(f"**{' · '.join(f'{k}: {v}' for k, v in c['key'].items())}**")
            for paper, vals in c["values_by_paper"].items():
                st.markdown(f"- {paper}: **{', '.join(vals)}**")

# ---------------- Leaderboard tab ----------------
with tab_board:
    verified = df[df["status"] == "verified"].copy()
    if verified.empty:
        st.info("No verified rows yet.")
    else:
        b1, b2, b3, b4 = st.columns(4)
        score_col = b1.selectbox("Score field", fields, index=len(fields) - 1)
        name_col = b2.selectbox("Rank by", [f for f in fields if f != score_col] or fields)
        group_opts = ["(none)"] + [f for f in fields if f not in (score_col, name_col)]
        # Default to a dataset/benchmark-like field so different datasets aren't mixed
        guess = next((i for i, f in enumerate(group_opts)
                      if any(k in f.lower() for k in ("dataset", "benchmark", "task", "data"))), 0)
        group_col = b3.selectbox("Separate by", group_opts, index=guess,
                                 help="e.g. Dataset, so CIFAR-10 and ImageNet aren't mixed")
        # Default to 'Lower' when most values look like error/loss
        vals = verified[score_col].astype(str).str.lower()
        lower_default = vals.str.contains("error|loss|wer|perplexity").mean() > 0.5
        higher = b4.radio("Better is", ["Higher", "Lower"], index=int(lower_default), horizontal=True,
                          help="Use 'Lower' for error rates or loss") == "Higher"

        verified["_score"] = verified[score_col].map(to_number)
        verified = verified.dropna(subset=["_score"])
        groups = [(None, verified)] if group_col == "(none)" else list(verified.groupby(group_col))
        if not groups:
            st.info(f"No numeric values found in '{score_col}'.")
        for name, g in groups:
            if name is not None:
                st.subheader(str(name))
            board = (g.sort_values("_score", ascending=not higher)
                      .drop_duplicates(subset=[name_col])
                      [[name_col, score_col, "paper", "_score"]])
            board.insert(0, "Rank", range(1, len(board) + 1))
            st.dataframe(board.drop(columns="_score"), hide_index=True, width="stretch")
            st.bar_chart(board.set_index(name_col)["_score"], horizontal=True)
        st.caption("Only ✅ verified rows are ranked. Each entry keeps its best score across papers.")

# ---------------- Summary tab ----------------
with tab_summary:
    verified_rows = df[df["status"] == "verified"].to_dict("records")
    excluded = len(df) - len(verified_rows)
    st.caption(f"Written only from the {len(verified_rows)} ✅ verified rows ({excluded} others excluded), "
               f"organized by your columns: {', '.join(fields)}.")
    focus = st.text_input("Focus (optional)", placeholder="ex) Claude, MMLU, mu1, or a column name like Parameter",
                          help="A value (model, benchmark, parameter like μ₁ / mu1, table) summarizes only "
                               "matching rows. A column name groups the summary by that column.")
    if st.button("Generate summary", disabled=not verified_rows):
        with st.spinner("Writing..."):
            try:
                ss.summary = summarize(verified_rows, fields, excluded, focus=focus or None)
            except Exception as e:
                st.error(f"Summary failed: {e}")
    s = ss.summary
    if isinstance(s, dict):
        if not s["text"]:
            st.info(f'No verified rows mention "{s.get("facts", {}).get("focus") or focus}". '
                    "Focus matches values in your results, or a column name "
                    f"({', '.join(fields)}) to group by.")
            if s.get("examples"):
                st.markdown("**Values you can focus on:**  \n" + "  \n".join(
                    f"*{col}:* " + ", ".join(f"`{v}`" for v in vals) for col, vals in s["examples"].items()))
        else:
            f = s["facts"]
            st.markdown(s["text"])
            how = []
            if f.get("group_column"):
                how.append(f"grouped by **{f['group_column']}**")
            if f.get("measure_column"):
                how.append(f"each **{f['measure_column']}** compared separately")
            if f.get("entity_column") and f.get("score_column"):
                how.append(f"comparing **{f['entity_column']}** by **{f['score_column']}**"
                           + (f" ({f['better_is']} is better)" if f.get("better_is") else ""))
            st.caption(f"Based on {s['rows_used']} verified rows" + (f", {', '.join(how)}" if how else "") +
                       (f', focused on "{f["focus"]}"' if f.get("focus") else "") + ".")
            if s["unsupported"]:
                st.warning(f"These numbers in the summary don't appear in the verified data: "
                           f"{', '.join(s['unsupported'])}. Treat them with caution.")
            with st.expander("What the summary is based on (computed by code)"):
                st.json(f)
