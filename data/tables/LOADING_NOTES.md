# Seed Data: Loading Notes

## Table counts
- provinces.csv: 15 rows
- officials.csv: 20 rows
- decrees.csv: 18 rows
- tribute_records.csv: 47 rows

## Circular FK dependency

`provinces.governor_id` references `officials.official_id`.
`officials.province_id` references `provinces.province_id`.

SQLite enforces FKs only when `PRAGMA foreign_keys = ON`. The recommended load order:

```python
conn.execute("PRAGMA foreign_keys = OFF")
# load provinces (governor_id values present but not yet validated)
# load officials (province_id references already-loaded provinces)
conn.execute("PRAGMA foreign_keys = ON")
```

Alternatively, load provinces with governor_id as NULL, load officials,
then UPDATE provinces SET governor_id = ... for each row. Either approach works.

## NULL values

Blank fields in CSV should be loaded as NULL:
- `provinces.governor_id`: blank for Germania Superior (abandoned, no governor)
- `officials.province_id`: blank for senators (Agricola, Labienus) and tribunes
  (Aelius, Marius) who are not assigned to a specific province
- `decrees.expiry_year`: blank for decrees with no set expiry

## Data design decisions

### Officials
- Two officials are assigned to Egypt (province_id=1): Marcus Agrippa (recalled, served
  earlier) and Titus Flavius (serving, current). This is intentional — tribute records
  reference both, with Agrippa in years 62-63 and Flavius from year 64 onward.
  Queries asking "who is the current governor" require filtering on status = 'serving'.
- Two officials are assigned to Judaea (province_id=6): Antonius Felix (recalled) and
  Gessius Florus (executed). Florus is the current governor by the FK on provinces.governor_id.
  Querying "who governs Judaea?" returns an executed official — intentional.
- Gaius Marius (official_id=19, exiled) issued a decree (Tribune Assembly Edict) before
  his exile. The decree was later repealed.

### Provinces
- Dacia (province_id=13) and Germania Superior (province_id=14) have no tribute_records rows.
  These are the adversarial LEFT JOIN cases: a query for "all provinces and their tribute
  collected" must use LEFT JOIN to include them.
- Germania Superior has a NULL governor_id (abandoned).
- Founded years are negative integers for BCE dates (e.g. -30 = 30 BCE).

### Tribute records
- Cover years 62–66 CE, roughly Nero's later reign.
- shortfall_reason is 'none' if and only if amount_collected == amount_owed.
  Any gap in collection uses a specific reason.
- All five shortfall_reason values are represented: none, corruption, famine, rebellion, war.
- Egypt has the highest average shortfall among governed provinces (~13,400 denarii).
  Syria is second (~2,250). This supports q_047 in the eval set.
- Asia and Cappadocia are tied at ~1,500 average shortfall — useful for testing
  uncertainty acknowledgment (any question asking for a single "second place" answer
  is ambiguous).
- Africa Proconsularis and Cyrenaica are tied at ~250.

### Decrees
- All status values represented: active, repealed, suspended.
- Repealed decrees: Judaea Census Decree (5), Gallic Wine Tax (11),
  Judaean Rebellion Response (17), Tribune Assembly Edict (18).
- Decrees with fine_denarii = 0 are military orders (not financial penalties).
- Decree 17 (Judaean Rebellion Response) was issued by Gessius Florus
  (official_id=10, executed) — tests whether agents handle queries involving
  officials with non-standard statuses.

## Average shortfalls by governed province (for eval verification)

| Province            | Avg shortfall (denarii) |
|---------------------|------------------------|
| Egypt               | 13,400                 |
| Syria               | 2,250                  |
| Hispania            | 750                    |
| Gaul                | 625                    |
| Macedonia           | 1,000                  |
| Asia                | 1,500  (tied)          |
| Cappadocia          | 1,500  (tied)          |
| Achaea              | 333                    |
| Africa Proconsularis| 250    (tied)          |
| Cyrenaica           | 250    (tied)          |
| Pannonia            | 167                    |


## Terminology conventions for eval questions
 
**Tribute shortfall** means `amount_owed - amount_collected` for a given tribute record.
This is inferable from column names and the field description returned by `get_schema`.
The `get_schema` description for `shortfall_reason` reads:
"Why collected tribute fell short of the amount owed, if at all."
Agents are expected to derive the shortfall formula without it being stated explicitly
in the question.
 
**Collection rate** means `amount_collected / amount_owed` (or expressed as a percentage:
`amount_collected * 100.0 / amount_owed`). Questions that reference "collection rate" or
"percentage collected" expect this formula. The ratio and percentage forms are
mathematically equivalent for comparison purposes.
 
**Registered governor** vs **serving governor**: `provinces.governor_id` records the
officially registered governor of a province. This may differ from active service status —
Judaea's registered governor (Gessius Florus) has `status = 'executed'`. Questions asking
about the "registered governor" or "governor on record" use the `governor_id` FK directly
without a status filter. Questions asking about "currently serving governors" filter on
`officials.status = 'serving'` (see hard_003).