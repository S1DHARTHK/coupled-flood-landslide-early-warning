# New data sources — 2026-09-11 correction task

Every dataset below was downloaded on **2026-09-11** (17:05–17:21 UTC) directly from the
official publisher. Originals are stored unchanged; exact URLs, byte counts and SHA-256
checksums are in `documentation/download_log.tsv` and `documentation/SOURCE_MANIFEST_2026-09-11.json`.
Processing scripts are in `updated_data/scripts/` (01–08) and regenerate every processed file.

---

## 1. CWC observed river water level — COLLECTED

| Field | Value |
|---|---|
| Dataset | River water level, CWC hydrological observation stations, Kerala |
| Source Organization | Central Water Commission (CWC), Ministry of Jal Shakti, Government of India — published by the National Water Informatics Centre (NWIC) |
| Official Dataset Name | River Water Level (Manual - Hourly), Central Water Commission (CWC) — basin "West flowing rivers from Tadri to Kanyakumari" (NWDP dataset code WQT00004) |
| Official Source | National Water Data Portal (NWDP), Government of India |
| URL | Dataset page: https://nwdp.nwic.gov.in/dataset/river-water-level-manual-hourly-cwc-subernarekha · files: `rwl_manual_hr_cwc_031_1991_2020.csv`, `rwl_manual_hr_cwc_031_2021_2025.csv` (full resource URLs in download_log.tsv) |
| Access Date | 2026-09-11 |
| Coverage | 31 CWC gauges in Kerala (plus Karnataka/Tamil Nadu gauges of the same basin, not used); Kerala readings 1991-01-01 → 2025-12-31. 5 gauges continuous 2012–2024; 10 more from mid-2015 |
| Resolution | Point gauges; manual readings (typically 08:00, 13:00, 18:00; hourly during floods). Aggregated to daily per gauge |
| Variables Used | `Station`, `District LGD Code`, `River`, `Latitude`, `Longitude`, `RL_of_zeroGauge`, `Data Acquisition Time`, `River Water Level Manual Hourly (meter)` |
| Units | metres (dataset metadata "Unit = m"); coordinates in decimal degrees (WGS84) |
| Why It Is Suitable | Observed river stage in metres, from the national authority named first in the brief, covering the entire training window (2012-01-30 → 2021-02-14, derived from `ml/flood_xgboost.py` `chronological_split()`). Data Access Control on the portal = **Public**; licence "Other (Open)". |
| Limitations | Point gauges, not district averages. Some readings are published as the gauge reading and others as gauge reading + RL of gauge zero (datum mixing) — handled by removal, never by conversion. 7 river gauges keep unexplained datum breaks (series split into segments). No CWC gauge in Alappuzha or Wayanad; Kottayam's only gauge starts 2019. Stage values from different gauges are on different references and are not directly comparable across districts. |

## 2. Kerala Surface Water Department manual river level — COLLECTED (supplementary only)

| Field | Value |
|---|---|
| Dataset | River water level, Kerala SW manual stations |
| Source Organization | Kerala Surface Water Department / Irrigation Design and Research Board (IDRB), Government of Kerala — published by NWIC |
| Official Dataset Name | River Water Level (Manual - Hourly), Kerala Surface Water Department — "West flowing rivers from Tadri to Kanyakumari" (NWDP dataset code WQT00019) |
| Official Source | National Water Data Portal (NWDP) |
| URL | https://nwdp.nwic.gov.in/dataset/river-water-level-manual-hourly-kerala-sw-godavari · file `rwl_manual_hr_kerala_sw_031_2021_2025.csv` |
| Access Date | 2026-09-11 |
| Coverage | 6 stations (Kannur, Kozhikode, Malappuram, Thrissur ×2, Wayanad); 2022-01-01 → 2025-03-31 |
| Resolution | Manual readings 3×/day |
| Variables Used | none in this task (inventory only) |
| Units | metres |
| Why It Is Suitable | Official observed stage; the only source found with a Wayanad station (from 2023) |
| Limitations | Starts 2022–2023, so no training-period data. River names in the file are wrong (e.g. "Luni", "Banas", "Sani" — portal metadata error). Not used for any processed output. |

## 3. IMD Disastrous Weather Events annual reports 2012–2024 — COLLECTED

| Field | Value |
|---|---|
| Dataset | Disastrous Weather Events (DWE) — annual official record of weather disasters, by state |
| Source Organization | India Meteorological Department (IMD), Climate Research & Services, Pune — Ministry of Earth Sciences |
| Official Dataset Name | "Disastrous Weather Events 2012" … "Disastrous Weather Events 2024" |
| Official Source | IMD Pune Library publications |
| URL | https://www.imdpune.gov.in/library/publication.html · files under https://www.imdpune.gov.in/library/public/ (13 PDFs; exact names in download_log.tsv) |
| Access Date | 2026-09-11 |
| Coverage | All India, calendar years 2012–2024; KERALA section of the "Floods and heavy rains" tables: 238 entries parsed (2013–2024) + 13 (2012, read visually) + 9 (2016, read manually) |
| Resolution | Event level: day or day-list / period; district names |
| Variables Used | Date/Period, Area affected (districts), Intensity, Casualties, Extent of damage |
| Units | dates; district names |
| Why It Is Suitable | IFI-Impacts v3.0 cites **IMD** as the source of all 608 Kerala records — these reports are the authoritative original against which every IFI date was verified. The same reports extend the event record past IFI's last Kerala event (2023-07-23) through 2024, and record dated landslide occurrences. |
| Limitations | DWE 2012 is a scanned image (no text layer) — Kerala pages were read visually. Some entries give periods (e.g. "29 May to 31 Jul.") rather than days. Multi-district entries do not say in which district a reported landslide occurred. Entries record damaging events reported to IMD; absence of an entry does not prove absence of flooding. |

## 4. Derived / corrected datasets produced in this task

| File | Built from | What it is |
|---|---|---|
| `river_level/processed/CWC_Kerala_RiverLevel_Daily_Station.csv` | CWC (1) | Daily mean/max/min/08:00 stage per gauge, datum-QC'd, with `datum_segment` |
| `river_level/processed/CWC_Kerala_station_inventory.csv` | CWC (1) | 31 gauges: location, official district, datum QC, coverage per split, training usability |
| `river_level/processed/District_Gauge_Assignment.csv` | CWC (1) | Which gauge may represent which district, and why |
| `flood_events/processed/IFI_Kerala_2012_2023_corrected.csv` | IFI v3.0 + IMD DWE (3) | All 288 Kerala IFI records (2012–2023), original columns unchanged + verified dates, IMD evidence, flags |
| `flood_events/processed/IFI_Kerala_date_corrections.csv` | same | The 47 records whose dates were corrected or whose event days are non-contiguous |
| `flood_events/processed/IMD_DWE_Kerala_events_not_in_IFI.csv` | IMD DWE (3) | 24 events after 2023-07-23 + 1 event omitted by IFI |
| `landslide_events/processed/GSI_Kerala_Landslides_date_precision.csv` | GSI inventory (existing) | Every GSI record classified EXACT_DATE / MONTH_YEAR / YEAR_ONLY / NO_DATE |
| `landslide_events/processed/IMD_DWE_Kerala_landslide_entries.csv` | IMD DWE (3) | 59 IMD entries reporting a landslide, with date precision and district attribution |
| `lithology/processed/Kerala_District_Lithology_CGWB_reproduced.csv` | CGWB zip (existing) | Independent reproduction of the existing lithology table (identical result) |

## 5. Sources checked but NOT obtained

| Source | Why not obtained | Manual step required |
|---|---|---|
| India-WRIS (indiawris.gov.in) | Host does not respond from this environment (connection timeout on every attempt) | Not needed any more for river level — CWC data was obtained via NWDP |
| GSI Bhukosh / NGDR (geology 1:50k, landslide inventory) | Hosts do not respond (timeout) | Register at https://bhukosh.gsi.gov.in, download the Kerala 1:50,000 lithology layer and the Kerala landslide inventory from a network that can reach the host |
| GSI Bhusanket `India_All_Landslided`, `GSI_Landslide_India` services | Return "Token Required" — access-controlled; not bypassed | Request access from GSI / use the Bhukosh download |
| GSI Bhusanket `Landslidedata_1` service | Public, but holds only 2 records nationally, none in Kerala, and contains personal contact fields | None — not relevant |
| KSDMA (sdma.kerala.gov.in) | Publishes narrative memoranda, government orders and a post-disaster needs assessment (PDFs) — no structured, dated event archive | None — IMD DWE covers the same events systematically |
| IFI-Impacts newer version | v3.0 (Zenodo 10.5281/zenodo.11275211) is the latest release; it ends 2023 | None |
