"""
Extract the KERALA entries of IMD "Disastrous Weather Events" (DWE) annual reports.

Input : flood_events/imd_dwe_source/*.pdf   (official IMD Pune library PDFs, unchanged)
Output: flood_events/processed/dwe_text/<report>.txt       (xpdf pdftotext -table output)
        flood_events/processed/IMD_DWE_Kerala_entries.csv   (one row per DWE table entry)

DWE tables have the columns: S.No | Date/Period | Area affected | Intensity | Casualties |
Extent of damage. An entry starts with "N. <day> ..." at the left margin. `districts` is read
from the Area-affected column only (column bounds taken from each page header); districts named
only in the casualty/damage text are kept separately in `districts_any_column`. The Date/Period
text is taken exactly as printed; `event_days` is expanded from it literally ("3 & 9 Sep."
stays two separate days, "9 to 13 Jun." becomes five days). Entries whose date text cannot
be parsed are flagged, never guessed. The raw text of every entry is kept for manual review.
"""
from pathlib import Path
import re
import subprocess
import sys
import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"
SRC = U / "flood_events" / "imd_dwe_source"
OUT = U / "flood_events" / "processed"
TXT = OUT / "dwe_text"
TXT.mkdir(parents=True, exist_ok=True)

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
SEP = r"(?:\s*(?:,|&|and|to|till|-|–)\s*)"
DATE_RE = re.compile(rf"^((?:\d{{1,2}}{SEP}?)+\s*(?:{MON})?(?:{SEP}(?:\d{{1,2}}{SEP}?)+\s*{MON})*)", re.I)
ENTRY_RE = re.compile(r"^\s{0,12}(\d{1,2})\.\s+(\d{1,2}\b.*)$")
LEAD_MON_RE = re.compile(rf"^\s{{0,16}}({MON})(?=\s|$)", re.I)

KERALA_DISTRICTS = ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam",
                    "Kottayam", "Kozhikode", "Malappuram", "Palakkad", "Pathanamthitta",
                    "Thiruvananthapuram", "Thrissur", "Wayanad"]
VARIANTS = {"thiruvananthpuram": "Thiruvananthapuram", "thiruvananthapurm": "Thiruvananthapuram", "thiruvanantha-": "Thiruvananthapuram",
            "trivandrum": "Thiruvananthapuram", "kasargod": "Kasaragod", "kasaragode": "Kasaragod",
            "pathanamthita": "Pathanamthitta", "palghat": "Palakkad", "calicut": "Kozhikode",
            "trichur": "Thrissur", "alleppey": "Alappuzha", "quilon": "Kollam",
            "cannanore": "Kannur", "idduki": "Idukki", "wayand": "Wayanad",
            "pattanamittia": "Pathanamthitta", "kozikode": "Kozhikode", "wanayad": "Wayanad"}
OTHER_STATES = ["KARNATAKA", "LAKSHADWEEP", "MADHYA PRADESH", "MAHARASHTRA", "TAMIL NADU",
                "PUDUCHERRY", "GOA", "LADAKH", "MANIPUR", "MEGHALAYA", "MIZORAM", "JHARKHAND"]
HEADER_RE = re.compile(r"S\.?\s*r?\.?\s*No?\.?\s+Date\s*/\s*Period", re.I)


def pdf_to_text(pdf: Path) -> Path:
    out = TXT / (pdf.stem + ".txt")
    if not out.exists() or out.stat().st_mtime < pdf.stat().st_mtime:
        subprocess.run(["pdftotext", "-table", str(pdf), str(out)], check=True)
    return out


def districts_in(text: str) -> list[str]:
    t = text.lower()
    found = {d for d in KERALA_DISTRICTS if d.lower() in t}
    found |= {v for k, v in VARIANTS.items() if k in t}
    if re.search(r"(all\s+)?districts\s+of\s+kerala|all\s+districts", t):
        found.add("ALL_DISTRICTS")
    return sorted(found)


def expand_days(date_text: str, year: int):
    """Literal expansion of an IMD Date/Period string. Returns (days, flag)."""
    toks = re.findall(rf"\d{{1,2}}|{MON}|to|till|-|–", date_text, re.I)
    items, pending, rng = [], [], False
    for t in toks:
        tl = t.lower().rstrip(".")
        if t.isdigit():
            pending.append((int(t), rng)); rng = False
        elif tl in ("to", "till", "-", "–"):
            rng = True
        elif tl[:3] in MONTHS:
            items += [(d, r, MONTHS[tl[:3]]) for d, r in pending]; pending = []
    if pending or not items:
        return [], "no_month"
    days = []
    try:
        for d, is_range_end, m in items:
            ts = pd.Timestamp(year, m, d)
            if is_range_end and days:
                if ts < days[-1]:
                    return [], "reversed_range"
                days += list(pd.date_range(days[-1] + pd.Timedelta(days=1), ts))
            else:
                days.append(ts)
    except ValueError:
        return [], "invalid_date"
    return sorted(set(days)), "ok"


STATE_TOKEN_RE = re.compile(r"\b(" + "|".join(OTHER_STATES) + r")\b")   # upper-case only


def kerala_sections(lines):
    """(start, end) line ranges of each KERALA state section in the detailed event tables.
    KERALA mentions before the first 'Date/Period' table header belong to the statistical
    summary tables at the front of the report and are ignored. A section ends at the next
    upper-case state name or at the '* Whole year' state summary."""
    first_table = next((i for i, l in enumerate(lines) if HEADER_RE.search(l)), None)
    if first_table is None:
        return []
    out = []
    for i, l in enumerate(lines):
        if i < first_table or not re.search(r"\bKERALA\b", l):
            continue
        j = i + 1
        while j < len(lines):
            s = lines[j]
            if re.match(r"^\s*\*\s*Whole year", s) or STATE_TOKEN_RE.search(s):
                break
            j += 1
        out.append((max(0, i - 4), j))
    return out


def parse_report(pdf: Path) -> list[dict]:
    year = int(re.search(r"(?:19|20)\d{2}", pdf.stem).group(0))
    raw = pdf_to_text(pdf).read_text(encoding="utf-8", errors="replace")
    lines = raw.split("\n")          # not splitlines(): that would also split on form feeds
    page, page_of = 1, []
    for l in lines:
        page += l.count("\f"); page_of.append(page)
    def int_cols(h):
        i_, c_ = h.find("Intensity"), h.find("Casualties")
        return (max(0, i_ - 6), c_ - 1) if i_ >= 0 and c_ > i_ else None

    def area_cols(h):
        a_, i_ = h.find("Area"), h.find("Intensity")
        # data rows can sit up to ~14 characters left of the printed header; district
        # names never collide with date or intensity words, so a generous window is safe
        return (max(0, a_ - 14), i_ + 4) if a_ >= 0 and i_ > a_ else None

    rows = []
    for a, b in kerala_sections(lines):
        cur, n_in_section = None, 0
        hdr = next((lines[h] for h in range(a, -1, -1) if HEADER_RE.search(lines[h])), "")
        cols, icols = area_cols(hdr), int_cols(hdr)
        for i in range(a, b):
            l = lines[i].replace("\f", "")
            if HEADER_RE.search(l):
                cols, icols = area_cols(l) or cols, int_cols(l) or icols
                continue                                # table header
            if re.fullmatch(r"\s*\d{1,3}\s*", l):
                continue                                # page number
            m = ENTRY_RE.match(l)
            wrapped = False
            if not m and cur is not None:
                # entry number printed alone, date on the following line(s): accept only
                # if it is exactly the next serial number and a date follows at the margin
                m2 = re.match(r"^\s{0,12}(\d{1,2})\.(?:\s+(.*))?$", l)
                if m2 and int(m2.group(1)) == cur["sno"] + 1:
                    for k in (1, 2):
                        nxt = lines[i + k].strip() if i + k < b else ""
                        if DATE_RE.match(nxt):
                            m = ENTRY_RE.match(f"{m2.group(1)}. {nxt}")
                            wrapped = True
                            break
            if m:
                if n_in_section == 0 and int(m.group(1)) != 1:
                    continue                            # tail of the previous state's table
                if int(m.group(1)) == 1 and n_in_section > 0:
                    break                               # numbering restarts: next state
                n_in_section += 1
                if cur: rows.append(cur)
                rest = m.group(2)
                dm = DATE_RE.match(rest)
                date_text = dm.group(1).strip() if dm else ""
                cur = dict(report=pdf.name, year=year, page=page_of[i], sno=int(m.group(1)),
                           date_text=date_text, line_no=i + 1,
                           lines=[rest[len(date_text):]] + ([l] if wrapped else []),
                           area=[l[cols[0]:cols[1]] if cols else ""],
                           inten=[l[icols[0]:icols[1]] if icols else ""])
                if not re.search(MON, date_text, re.I):   # month printed on the next line
                    for k in (1, 2):
                        if i + k < b:
                            lm = LEAD_MON_RE.match(lines[i + k])
                            if lm:
                                cur["date_text"] = (date_text + " " + lm.group(1)).strip(); break
                continue
            if cur is not None:
                cur["lines"].append(l)
                cur["area"].append(l[cols[0]:cols[1]] if cols else "")
                cur["inten"].append(l[icols[0]:icols[1]] if icols else "")
        if cur: rows.append(cur)
    out, seen = [], set()
    for r in rows:
        key = (r["year"], r["line_no"])
        if key in seen:
            continue
        seen.add(key)
        text = re.sub(r"\s+", " ", " ".join(r.pop("lines"))).strip()
        area = re.sub(r"\s+", " ", " ".join(r.pop("area"))).strip()
        inten = re.sub(r"\s+", " ", " ".join(r.pop("inten"))).replace("[River(s)]", "").strip()
        days, flag = expand_days(r["date_text"], r["year"])
        r.update(entry_text=text,
                 event_days=";".join(d.strftime("%Y-%m-%d") for d in days),
                 n_event_days=len(days), date_parse_flag=flag,
                 area_text=area, intensity_text=inten,
                 districts=";".join(districts_in(area) or districts_in(text)),
                 district_source=("area_column" if districts_in(area) else
                                  "any_column (area column not readable)" if districts_in(text) else "none"),
                 districts_any_column=";".join(districts_in(text)),
                 mentions_flood=bool(re.search(r"flood|inundat|submerg|water.?logg", text, re.I)),
                 mentions_landslide=bool(re.search(r"land\s?slide|land\s?slip|mud\s?slide", text, re.I)))
        out.append(r)
    return out


def main():
    only = sys.argv[1:]            # optional: restrict to reports whose name contains these
    allrows = []
    for pdf in sorted(SRC.glob("*.pdf")):
        if only and not any(o in pdf.name for o in only):
            continue
        rows = parse_report(pdf)
        chars = len(pdf_to_text(pdf).read_text(encoding="utf-8", errors="replace").strip())
        print(f"{pdf.name}: {len(rows)} Kerala entries (extracted text: {chars:,} chars)")
        allrows += rows
    df = pd.DataFrame(allrows)
    if df.empty:
        print("no Kerala entries parsed"); return
    df.to_csv(OUT / "IMD_DWE_Kerala_entries.csv", index=False)
    print(df.groupby(["year", "date_parse_flag"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
