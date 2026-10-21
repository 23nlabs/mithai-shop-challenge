# Festival calendar provenance (Mithai Shop Challenge)

Primary: `data/festivals.json` in this repo (officeholidays.com India country pages 2015–2027, per-record URL + page hash).
Diwali (Lakshmi Puja) cross-check: drikpanchang.com Lakshmi Puja timings page (dates computed for New Delhi), fetched 2026-10-21:

| Year | officeholidays primary (candidates) | drikpanchang | Used |
|---|---|---|---|
| 2019 | 10-27 (10-27, 10-28) | Oct 27 | 2019-10-27 |
| 2020 | 11-14 | Nov 14 | 2020-11-14 |
| 2021 | 11-04 | Nov 4 | 2021-11-04 |
| 2022 | 10-24 | Oct 24 | 2022-10-24 |
| 2023 | 11-12 (11-12, 11-13) | Nov 12 | 2023-11-12 |
| 2024 | 10-31 (10-31, 11-01) | **Nov 1** | **2024-11-01** (drikpanchang; regional observance varied) |
| 2025 | 10-20 (10-20, 10-21) | Oct 20 | 2025-10-20 |
| 2026 | 11-08 (11-08, 11-09; timeanddate agrees) | Nov 8 | 2026-11-08 |

Diwali 2025 → 2026 shift: +19 days. 2022 → 2023: +19 days. 2023 → 2024: −11. 2024 → 2025: −12.
Other festivals (Dussehra, Holi, Raksha Bandhan, Ganesh Chaturthi) use the officeholidays primary date
(single source except 2026 Raksha Bandhan, cross-checked with timeanddate). Navratri is derived as the nine
days ending the day before Dussehra. Wedding seasons are fixed Gregorian approximations (documented in
`src/mithai_core.py`); real muhurat dates vary by year and community.
