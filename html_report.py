"""A small dependency-free HTML report builder, shared by the EDA and results reports.

Produces one self-contained file: inline CSS, every figure embedded as a base64
PNG, a sticky table of contents, a KPI strip and zebra tables. No Jinja, no
external assets, opens offline.

Author: Rahul Jain
Date:   17 Nov 2026
Usage:  from ctg.html_report import Report
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from . import config as C

CSS = """
:root{--ink:#1a1d23;--muted:#5b6472;--line:#e3e7ee;--bg:#f6f7fa;--card:#ffffff;
--accent:#2f5d9e;--ok:#1d7a4d;--warn:#9a6400;--bad:#a81f2b;
--ok-bg:#e6f4ec;--warn-bg:#fdf3dd;--bad-bg:#fbe9ea;--info-bg:#eaf1fa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
a{color:var(--accent)}
.wrap{max-width:1200px;margin:0 auto;padding:0 24px 80px;}
header.top{background:linear-gradient(135deg,#1f3f6e,#2f5d9e);color:#fff;padding:28px 0 24px;
margin-bottom:24px;}
header.top .wrap{padding-bottom:0}
header.top h1{margin:0 0 6px;font-size:27px;letter-spacing:-.3px}
header.top .sub{opacity:.92;font-size:15px;margin-bottom:12px}
header.top .meta{font-size:12.5px;opacity:.85;line-height:1.8}
.badge{display:inline-block;background:#fff;color:#a81f2b;font-weight:700;font-size:11.5px;
letter-spacing:.06em;text-transform:uppercase;padding:4px 10px;border-radius:4px;margin-bottom:10px}
.layout{display:grid;grid-template-columns:238px minmax(0,1fr);gap:28px;align-items:start}
nav.toc{position:sticky;top:18px;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:14px 16px;font-size:13.5px;max-height:calc(100vh - 40px);overflow:auto}
nav.toc h3{margin:0 0 8px;font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;
color:var(--muted)}
nav.toc a{display:block;padding:4px 0;color:var(--ink);text-decoration:none;border-bottom:1px solid #f1f3f7}
nav.toc a:last-child{border-bottom:0}
nav.toc a:hover{color:var(--accent)}
section.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:22px 26px;margin-bottom:22px;scroll-margin-top:16px}
section.card h2{margin:0 0 4px;font-size:21px;letter-spacing:-.2px}
section.card h3{margin:26px 0 8px;font-size:16.5px}
section.card h4{margin:18px 0 6px;font-size:14.5px;color:var(--muted)}
.intro{color:var(--muted);margin:0 0 18px;font-size:14.5px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:4px 0 20px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:13px 15px}
.kpi .label{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.kpi .value{font-size:25px;font-weight:650;margin:3px 0 1px;letter-spacing:-.5px}
.kpi .note{font-size:12px;color:var(--muted)}
figure{margin:18px 0 22px}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:13.5px;margin-top:8px}
figcaption .cap{font-weight:600}
.howto{color:var(--muted);font-size:13px;margin-top:3px;font-style:italic}
.tablewrap{overflow-x:auto;margin:14px 0 20px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:#eef1f6;font-weight:650;position:sticky;top:0}
tbody tr:nth-child(even){background:#fafbfd}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.warn{background:var(--warn-bg)}
td.bad{background:var(--bad-bg)}
td.ok{background:var(--ok-bg)}
.tcap{font-size:13.5px;font-weight:600;margin-bottom:6px}
.callout{border-left:4px solid var(--accent);background:var(--info-bg);padding:11px 15px;
border-radius:0 6px 6px 0;margin:14px 0;font-size:14px}
.callout.warn{border-color:#c98a00;background:var(--warn-bg)}
.callout.bad{border-color:var(--bad);background:var(--bad-bg)}
.callout.ok{border-color:var(--ok);background:var(--ok-bg)}
.callout b{display:block;margin-bottom:2px}
.tag{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.05em;
padding:2px 7px;border-radius:3px;margin-right:5px;text-transform:uppercase}
.tag.pred{background:var(--ok-bg);color:var(--ok)}
.tag.out{background:var(--bad-bg);color:var(--bad)}
.tag.excl{background:#eceef2;color:var(--muted)}
.tag.meta{background:var(--info-bg);color:var(--accent)}
.tag.flag{background:var(--warn-bg);color:var(--warn)}
.varcard{border:1px solid var(--line);border-radius:7px;padding:16px 18px;margin:16px 0;
background:#fcfdff}
.varcard h3{margin:0 0 3px;font-size:17px}
.reading{background:#f2f5fa;border-radius:6px;padding:9px 13px;margin-top:10px;font-size:13.5px}
.reading b{color:var(--accent)}
footer{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;
margin-top:30px}
@media print{nav.toc{display:none}.layout{grid-template-columns:1fr}
section.card{break-inside:avoid;border:0;padding:0}header.top{background:#1f3f6e}}
@media(max-width:900px){.layout{grid-template-columns:1fr}nav.toc{position:static;max-height:none}}
"""

HighlightRule = Callable[[str, object], str]


def _fig_to_base64(fig, dpi: int = C.FIG_DPI_HTML) -> str:
    """Render a matplotlib figure to a base64 PNG string."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _file_to_base64(path: Path) -> str:
    """Read a PNG from disk as a base64 string."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _format_cell(value: object) -> str:
    """Format one table cell: numbers get tabular alignment, NaN becomes a dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:,.3g}" if abs(value) < 0.01 else f"{value:,.3f}"
    return html.escape(str(value))


class Report:
    """Accumulates sections, figures and tables, then writes one self-contained HTML."""

    def __init__(self, title: str, subtitle: str = "", run_name: str = "",
                 data_source: str = "", script_name: str = "",
                 figure_dir: Optional[Path] = None) -> None:
        self.title = title
        self.subtitle = subtitle
        self.run_name = run_name
        self.data_source = data_source
        self.script_name = script_name
        self.figure_dir = figure_dir
        self.created = datetime.now()
        self._toc: List[Tuple[str, str]] = []
        self._body: List[str] = []
        self._kpi_html = ""
        self._top_callouts: List[str] = []
        self._open = False
        self.n_figures = 0
        self.figure_paths: List[Path] = []

    # ------------------------------------------------------------- structure
    def add_section(self, section_id: str, heading: str, intro_text: str = "") -> "Report":
        """Start a new card section and register it in the table of contents."""
        if self._open:
            self._body.append("</section>")
        self._toc.append((section_id, heading))
        self._body.append(f'<section class="card" id="{html.escape(section_id)}">'
                          f'<h2>{html.escape(heading)}</h2>')
        if intro_text:
            self._body.append(f'<p class="intro">{intro_text}</p>')
        self._open = True
        return self

    def add_html(self, raw: str) -> "Report":
        """Append raw HTML (used for sub-headings and variable cards)."""
        self._body.append(raw)
        return self

    def add_subheading(self, text: str, level: int = 3) -> "Report":
        """Append an h3/h4 inside the current section."""
        return self.add_html(f"<h{level}>{html.escape(text)}</h{level}>")

    # ---------------------------------------------------------------- content
    def add_kpis(self, items: Sequence[Tuple[str, object, str]]) -> "Report":
        """Set the KPI strip shown under the header: (label, value, note) triples."""
        cells = "".join(
            f'<div class="kpi"><div class="label">{html.escape(str(label))}</div>'
            f'<div class="value">{html.escape(str(value))}</div>'
            f'<div class="note">{html.escape(str(note))}</div></div>'
            for label, value, note in items)
        self._kpi_html = f'<div class="kpis">{cells}</div>'
        return self

    def add_figure(self, fig_or_path, caption: str, how_to_read: str = "",
                   slug: str = "") -> "Report":
        """Embed a figure; a matplotlib figure is also written to the figure dir."""
        if hasattr(fig_or_path, "savefig"):
            encoded = _fig_to_base64(fig_or_path)
            if self.figure_dir is not None and slug:
                self.figure_dir.mkdir(parents=True, exist_ok=True)
                path = self.figure_dir / f"{slug}_{self.run_name}.png"
                fig_or_path.savefig(path, dpi=C.FIG_DPI_FILE, bbox_inches="tight")
                self.figure_paths.append(path)
        else:
            encoded = _file_to_base64(Path(fig_or_path))

        self.n_figures += 1
        howto = (f'<div class="howto">How to read: {html.escape(how_to_read)}</div>'
                 if how_to_read else "")
        self._body.append(
            f'<figure><img alt="{html.escape(caption)}" '
            f'src="data:image/png;base64,{encoded}">'
            f'<figcaption><span class="cap">Figure {self.n_figures}.</span> '
            f'{caption}{howto}</figcaption></figure>')
        return self

    def add_table(self, frame: pd.DataFrame, caption: str = "",
                  highlight_rules: Optional[Dict[str, HighlightRule]] = None,
                  index: bool = False, max_rows: Optional[int] = None) -> "Report":
        """Render a dataframe as a zebra table; highlight_rules shade cells."""
        data = frame if max_rows is None else frame.head(max_rows)
        if index:
            data = data.reset_index()

        header = "".join(f"<th>{html.escape(str(col))}</th>" for col in data.columns)
        rows = []
        for _, row in data.iterrows():
            cells = []
            for col in data.columns:
                value = row[col]
                css = "num" if isinstance(value, (int, float)) and not isinstance(
                    value, bool) else ""
                if highlight_rules and col in highlight_rules:
                    flag = highlight_rules[col](col, value)
                    if flag:
                        css = f"{css} {flag}".strip()
                attr = f' class="{css}"' if css else ""
                cells.append(f"<td{attr}>{_format_cell(value)}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")

        cap = f'<div class="tcap">{caption}</div>' if caption else ""
        note = ""
        if max_rows is not None and len(frame) > max_rows:
            note = (f'<div class="howto">Showing {max_rows} of {len(frame)} rows; '
                    "the full table is in output/sheets/.</div>")
        self._body.append(f'{cap}<div class="tablewrap"><table><thead><tr>{header}'
                          f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{note}")
        return self

    def add_callout(self, text: str, kind: str = "info", title: str = "") -> "Report":
        """Add a coloured callout box: info, warn, bad or ok."""
        css = "" if kind == "info" else f" {kind}"
        head = f"<b>{html.escape(title)}</b>" if title else ""
        self._body.append(f'<div class="callout{css}">{head}{text}</div>')
        return self

    def add_top_callout(self, text: str, kind: str = "info", title: str = "") -> "Report":
        """Add a callout that appears at the very top, above the first section."""
        css = "" if kind == "info" else f" {kind}"
        head = f"<b>{html.escape(title)}</b>" if title else ""
        self._top_callouts.append(f'<div class="callout{css}">{head}{text}</div>')
        return self

    # ------------------------------------------------------------------ save
    def save(self, path: Path, versions: Optional[Dict[str, str]] = None) -> Path:
        """Write the self-contained HTML file and return its path."""
        if self._open:
            self._body.append("</section>")
            self._open = False

        toc = "".join(f'<a href="#{sid}">{html.escape(heading)}</a>'
                      for sid, heading in self._toc)
        version_line = ", ".join(f"{k} {v}" for k, v in (versions or {}).items())
        stamp = self.created.strftime("%d %b %Y, %H:%M")

        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(self.title)}</title><style>{CSS}</style></head><body>
<header class="top"><div class="wrap">
<div class="badge">{html.escape(C.NOT_FOR_CLINICAL_USE)}</div>
<h1>{html.escape(self.title)}</h1>
<div class="sub">{html.escape(self.subtitle)}</div>
<div class="meta">Run <b>{html.escape(self.run_name)}</b> &middot; generated {stamp}<br>
Data source: {html.escape(self.data_source)}</div>
</div></header>
<div class="wrap">
{self._kpi_html}
{''.join(self._top_callouts)}
<div class="layout">
<nav class="toc"><h3>Contents</h3>{toc}</nav>
<main>{''.join(self._body)}
<footer>Generated by <code>{html.escape(self.script_name)}</code> on
{stamp} &middot; {self.n_figures} figures &middot; {html.escape(version_line)}<br>
Aggregates only: no participant-level data appears in this report.
{html.escape(C.NOT_FOR_CLINICAL_USE)}.</footer>
</main></div></div></body></html>"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
        return path


# ------------------------------------------------------------ highlight rules
def flag_above(threshold: float, kind: str = "warn") -> HighlightRule:
    """Shade a numeric cell when it exceeds a threshold."""
    def rule(_column: str, value: object) -> str:
        try:
            return kind if float(value) > threshold else ""
        except (TypeError, ValueError):
            return ""
    return rule


def flag_below(threshold: float, kind: str = "warn") -> HighlightRule:
    """Shade a numeric cell when it falls below a threshold."""
    def rule(_column: str, value: object) -> str:
        try:
            return kind if float(value) < threshold else ""
        except (TypeError, ValueError):
            return ""
    return rule


def flag_contains(needles: Iterable[str], kind: str = "warn") -> HighlightRule:
    """Shade a text cell when it contains any of the given substrings."""
    needles = list(needles)
    def rule(_column: str, value: object) -> str:
        text = str(value)
        return kind if any(needle in text for needle in needles) else ""
    return rule


def role_tag(role: str) -> str:
    """Return the coloured tag HTML for a variable role."""
    key = {"Predictor": "pred", "Outcome": "out",
           "Excluded": "excl", "Metadata": "meta"}.get(role.split(" ")[0], "excl")
    return f'<span class="tag {key}">{html.escape(role)}</span>'
