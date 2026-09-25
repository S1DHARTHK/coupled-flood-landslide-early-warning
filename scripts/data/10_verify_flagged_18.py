"""
Final, independent verification of the 18 flagged Kerala flood records of IFI-Impacts v3.0.

Independence from the earlier check (03_verify_ifi_kerala_dates.py): that step matched IFI to
IMD by DATE. This step locates each record's source entry by CONTENT - distinctive phrases of
IFI's own "Description of Casualties/injured" and "Extent of damage" fields, which IFI
digitised from the IMD Disastrous Weather Events (DWE) reports - and only then reads the date
IMD printed for that entry. A date is never inferred; it is read from the source entry.

Inputs (unchanged): hazard_events/India_Flood_Inventory_v3.csv
                    flood_events/imd_dwe_source/*.pdf (text layer via xpdf pdftotext -table,
                    cached in flood_events/processed/dwe_text/)
Outputs: flood_events/verification_18/content_match_candidates.csv  (automatic phrase hits)
         flood_events/verification_18/excerpts/<UEI>.txt            (source excerpt per record)
"""
from pathlib import Path
import re
import pandas as pd

U = Path(__file__).resolve().parents[2] / "collected_datasets"
TXT = U / "flood_events" / "processed" / "dwe_text"
OUT = U / "flood_events" / "verification_18"
(OUT / "excerpts").mkdir(parents=True, exist_ok=True)
REPORT = {2017: "DWE-2017_final", 2018: "DWE-2018_final", 2019: "DWE-2019", 2020: "DWE-2020", 2021: "DWE_2021"}
ENTRY = re.compile(r"^\s{0,12}(\d{1,2})\.\s+(\S.*)$")
STOP = {"reported", "damage", "to", "the", "of", "and", "in", "a", "an", "due", "several", "many"}


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()


def flagged():
    f = pd.read_csv(U / "hazard_events" / "India_Flood_Inventory_v3.csv", low_memory=False)
    k = f[f["State"].astype(str).str.contains("Kerala", case=False, na=False)].copy()
    sd = pd.to_datetime(k["Start Date"], format="%d-%m-%Y %H:%M")
    ed = pd.to_datetime(k["End Date"], format="%d-%m-%Y %H:%M")
    return k[((ed - sd).dt.days + 1 - k["Duration(Days)"]).abs() > 1]


def kerala_section(lines):
    first_tab = next(i for i, l in enumerate(lines) if re.search(r"Date\s*/\s*Period", l))
    s = next(i for i, l in enumerate(lines) if i > first_tab and re.search(r"\bKERALA\b", l))
    e = next(i for i, l in enumerate(lines) if i > s and re.search(r"\b(MADHYA|MAHARASHTRA|LAKSHADWEEP|MANIPUR)\b", l))
    return max(0, s - 3), e


def phrases(r):
    txt = f"{r['Description of Casualties/injured']} ; {r['Extent of damage ']}"
    clauses = re.split(r"\b[ivx]+\)\s*|;|\.\s", txt)
    out = []
    for c in clauses:
        w = [x for x in norm(c).split() if x != "nan"]
        if len(w) >= 3:
            out += [" ".join(w[i:i + 5]) for i in range(0, max(1, len(w) - 4))]
    return [p for p in dict.fromkeys(out) if len(set(p.split()) - STOP) >= 2]


def main():
    rows = []
    for _, r in flagged().iterrows():
        y = int(r["UEI"][11:15])
        lines = (TXT / f"{REPORT[y]}.txt").read_text(encoding="utf-8", errors="replace").split("\n")
        a, b = kerala_section(lines)
        page = lambda i: 1 + sum(l.count("\f") for l in lines[:i])
        # entry headers inside the Kerala section
        heads = [(i, int(ENTRY.match(lines[i]).group(1)), ENTRY.match(lines[i]).group(2))
                 for i in range(a, b) if ENTRY.match(lines[i]) and re.match(r"\d", ENTRY.match(lines[i]).group(2).strip())]
        normline = [norm(l) for l in lines]
        # entry text blocks (all columns) for every Kerala entry
        starts = [x[0] for x in heads] + [b]
        block = {heads[j][1]: " ".join(normline[starts[j]:starts[j + 1]]) for j in range(len(heads))}
        # evidence = 5-word phrases + rare single words (e.g. place names) from IFI's own text;
        # only evidence found in exactly ONE Kerala entry of that year's report is counted
        words = {w for p_ in phrases(r) for w in p_.split() if len(w) >= 6 and w not in STOP
                 and not re.fullmatch(r"\d+", w)}
        hits = {}
        for ev in list(phrases(r)) + sorted(words):
            where = [sno for sno, txt in block.items() if re.search(rf"\b{re.escape(ev)}\b", txt)]
            if len(where) == 1:
                hits.setdefault(where[0], set()).add(ev)
        best = sorted(hits.items(), key=lambda kv: -len(kv[1]))
        cand = best[0] if best else (None, set())
        h = next((x for x in heads if x[1] == cand[0]), None)
        rows.append(dict(UEI=r["UEI"], ifi_start=r["Start Date"][:10], ifi_end=r["End Date"][:10],
                         ifi_duration=r["Duration(Days)"], ifi_districts=r["Districts"],
                         report=REPORT[y] + ".pdf", n_phrases_tested=len(phrases(r)),
                         matched_entry=cand[0], n_phrases_matched=len(cand[1]),
                         other_entries_matched=";".join(f"{k}:{len(v)}" for k, v in best[1:]),
                         entry_header_text=(h[2].strip() if h else ""), entry_pdf_page=(page(h[0]) if h else ""),
                         matched_phrases=" | ".join(sorted(cand[1]))[:400]))
        if h:
            nxt = min([x[0] for x in heads if x[0] > h[0]] + [b])
            ex = "\n".join(lines[max(a, h[0] - 1):min(nxt + 1, h[0] + 40)])
            (OUT / "excerpts" / f"{r['UEI']}.txt").write_text(
                f"Source: IMD Disastrous Weather Events {y} ({REPORT[y]}.pdf), PDF page {page(h[0])}, KERALA entry {h[1]}\n"
                f"IFI record: {r['UEI']} | Start {r['Start Date']} | End {r['End Date']} | Duration {r['Duration(Days)']} | "
                f"Districts {r['Districts']}\n{'-' * 100}\n{ex}\n", encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "content_match_candidates.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(df[["UEI", "ifi_start", "ifi_end", "matched_entry", "n_phrases_matched", "n_phrases_tested",
              "other_entries_matched", "entry_header_text"]].to_string(index=False))


if __name__ == "__main__":
    main()
