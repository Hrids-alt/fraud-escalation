# FraudFin web app: design spec (paste this whole file into your AI tool)

You are building a small static web app for a finished machine-learning project. All results already exist. The app only **displays** them. It must not compute new model results, and it must not invent numbers.

## 1. What the project is (one paragraph, for the landing page)

FraudFin is a two-stage detector for illicit Bitcoin transactions (Elliptic dataset). Stage 1 is a cheap check that uses only each transaction's own features. Only the transactions Stage 1 cannot confidently clear are escalated to Stage 2, a more expensive check that also uses neighbour (graph) features. The app shows, on held-out time steps 35-49, how much detection quality the cascade keeps and how much simulated cost it saves compared with always running the expensive check.

## 2. Hard constraints

- Plain HTML, CSS and JavaScript. **No backend, no build step, no framework.**
- Files: `web/index.html`, `web/styles.css`, `web/app.js`, and the existing `web/data/results.json` (do not modify it).
- One chart library only: Chart.js 4.4.1 from cdnjs (`https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js`). No other external scripts. System fonts only.
- Load data with `fetch("data/results.json")` (relative path, so it works on GitHub Pages). If the fetch fails (for example the page was opened from `file://`), show a clear message: "Run `python -m http.server` inside the `web` folder and open http://localhost:8000".
- Works at 360 px wide and on desktop. Dark and light themes (follow `prefers-color-scheme`, plus a toggle). Wrap any `localStorage` use in try/catch and work without it.
- Accessible: keyboard operable, visible focus, contrast AA, never colour alone (use labels or icons too).
- **Do not copy any other product's layout, branding or text.** Use the working name "FraudFin" and the original tagline "Check cheap first. Look closer only when it matters." (the user may change both).

## 3. Visual direction

Calm financial-analytics look: card-based layout, one accent colour (teal or deep blue), neutral greys, generous spacing, large readable numbers, small muted captions. Tables with clear row separation. Rounded corners (8-12 px). No gradients on text, no decorative stock imagery, no emoji in the UI.

Status colours (always with a text label): caught = accent, false alarm = amber, missed = red, correctly cleared = grey.

## 4. Data contract: `web/data/results.json`

All numbers are pre-rounded to 5 decimals. `null` means "not available". Top-level keys and exact shape:

```
meta: {
  title, family: "rf",
  thresholds: { d: 0.39, d2: 0.26, t_clear: 0.39 },
  windows: { stage1_trained_on:[1,28], calibrated_on:[29,34],
             stage2_final_trained_on:[1,34], test:[35,49] },
  n_test: 16670, cost_label: "simulated",
  dataset: "Elliptic Bitcoin dataset (CC BY-NC-ND 4.0)",
  recall_retention_target: 0.95 }

kpis: { transactions: 16670, illicit_transactions: 1083, cleared_at_stage1: 15538,
        escalated: 1132, escalation_rate: 0.06791, validation_escalation_rate: 0.19522,
        illicit_caught_cascade: 780, illicit_caught_graph_always: 795,
        recall_cascade: 0.72022, recall_graph_always: 0.73407, recall_retention: 0.98113,
        precision_cascade: 0.8945, f1_cascade: 0.79795 }

systems: [ { system, label, escalation_rate, precision, recall, f1, pr_auc,
             recall_retention, tp, fp, fn, std? } ]
   // system is one of: local_only, graph_always, cascade, random_control_mean
   // tp/fp/fn and pr_auc can be null; random_control_mean has std:{precision,recall,f1}

cost: { formula, escalation_rate: 0.06791, r_values: [5,20,100],
        rows: [ { r, graph_always_cost, cascade_cost, relative_savings } ],
        slider: { min: 5, max: 100 }, label: "simulated" }

curve: { points: [ { t_clear, escalation_rate, cascade_precision, cascade_recall,
                     cascade_f1, recall_retention, random_recall_mean, random_f1_mean } ],  // 40 points
         local_only_recall, graph_always_recall,
         frozen_point: { ...same fields as a point }, note }

per_step: { local_only: [ { time_step, precision, recall, f1 } ],   // 15 rows, steps 35-49
            graph_always: [...], cascade: [...], random_control_mean: [...] }
   // precision/recall/f1 can be null where undefined: draw a gap, never 0

batches: [ { time_step, n, n_illicit, cleared, escalated, caught_cascade,
             false_alarms_cascade, missed_cascade, caught_graph_always, caught_local_only } ]  // 15 rows

queue: [ { txId, time_step, p1, escalated: bool, p2 (null if cleared), final_decision: 0|1,
           label: 0|1, local_only_decision, graph_always_decision,
           outcome: "caught"|"false alarm"|"missed"|"correctly cleared", why } ]  // 1,282 rows

seed_spread: [ { family, seed: "42"|"43"|"44"|"mean"|"std", system, escalation_rate,
                 precision, recall, f1, recall_retention } ]  // 20 rows; may be null

honesty: [ string, ... ]   // 9 lines; display them verbatim
```

**Important:** `queue` contains every escalated transaction plus only a **sample** of cleared ones. Never compute totals from the queue. Totals always come from `batches` (or `kpis` for "All").

## 5. Screens

### 5.1 Landing page
Product name, tagline, the paragraph from section 1, a three-step visual (Stage 1 cheap check -> escalate only if unsure -> Stage 2 final decision), and a **Start a batch** button that opens the dashboard. A short line under it: "Results on held-out time steps 35-49 of the Elliptic dataset. Cost is simulated."

### 5.2 Dashboard
A header with the name, a theme toggle and a "How to read this" button (opens the honesty panel, section 6).

**Batch picker.** A select: "All test steps (35-49)" or "Time step N" for each of the 15 steps. Changing it updates the KPI cards, the cost card and the queue. (The charts always show all steps; highlight the chosen step on the per-step chart.)

**KPI cards** (from `batches`, summed for "All"; for "All" they must equal `kpis`):
- Transactions = sum n
- Cleared at Stage 1 = sum cleared
- Escalated = sum escalated, with escalation rate = escalated / n (percentage)
- Illicit caught (cascade) = sum caught_cascade, shown as "X of Y illicit" (Y = sum n_illicit)
- Recall kept vs graph-always = sum caught_cascade / sum caught_graph_always (show "n/a" if the denominator is 0)
- Simulated cost saved = from the cost card below
- Also show smaller: false alarms (cascade) and missed (cascade).

**Simulated cost card.** A slider for r from 5 to 100 (step 1, default 20), with tick labels at 5, 20 and 100. With e = current escalation rate for the selected batch:
`graph_always = 1 + r`, `cascade = 1 + e * r`, `savings = 1 - cascade / graph_always`.
Show the three numbers, a two-bar comparison, and the label "simulated: the dataset's graph features are precomputed, so cost is modelled as local = 1 and graph = r times local". The formula text from `cost.formula` must be visible.

**Systems table.** One row per system from `systems` (use the `label`), columns: escalation rate, precision, recall, F1, PR-AUC, recall retention. Show "n/a" for null. Show the random-control std as "+/- x" under its value when present. Highlight the cascade row.

**Cost-versus-recall chart** (Chart.js scatter/line):
- x = escalation rate (0 to 1), y = recall.
- Cascade line from `curve.points` (cascade_recall), random escalation line (random_recall_mean), dotted horizontal lines for `local_only_recall` and `graph_always_recall`.
- A ringed marker at `curve.frozen_point` labelled "Frozen t_clear = 0.39".
- The y-axis starts near 0.72, so print a visible note under the chart: "Y-axis is zoomed: all recall differences here are about 1.4 percentage points or less."
- One-line takeaway: "The frozen threshold sits at the cheapest point: about 7% escalation keeps about 98% of graph-always recall. The curve is descriptive; the threshold was not chosen from it."

**Per-step chart** (line): F1 by time step for the four systems; null = gap. Shade steps 43-49 and label it "Drift: all systems collapse after step 42". Takeaway: "From step 43 every system fails; the cascade does not fix temporal drift. Late steps have very few illicit transactions, so single-step values are noisy."

**Seed-spread panel** (if `seed_spread` is not null): a small table of the `mean` and `std` rows for cascade and graph_always (F1, recall, precision, retention), with the caption "Final Stage 2 retrained with 3 seeds; thresholds unchanged; only Stage 2 varies."

**Escalation queue** (table, scrollable, paginated or virtualised, 25 rows per page):
- Rows from `queue`, filtered by the selected batch step.
- Filter chips: All, Escalated, Cleared (sample), Caught, False alarm, Missed.
- Columns: Transaction id, step, p1 (3 decimals), route (Escalated / Cleared), p2 (3 decimals or "-"), final decision (Illicit / Licit), true label, outcome (coloured label with text).
- Caption: "All escalated transactions plus a sample of cleared ones. Ids are the dataset's own anonymised ids."
- Clicking a row opens a **"How this was decided"** panel with these steps, using the real values and `meta.thresholds`:
  1. Stage 1 (local features) gave p1 = ...
  2. Routing: p1 vs t_clear (0.39) -> cleared as licit, or escalated.
  3. If escalated: Stage 2 (adds graph features) gave p2 = ... vs d2 (0.26) -> illicit or licit.
  4. Final decision.
  5. True label from the dataset, and the outcome (caught / false alarm / missed / correctly cleared).
  6. For comparison: what local-only and graph-always would have decided (`local_only_decision`, `graph_always_decision`).
  Use the `why` field as the one-line summary. The panel is a side drawer on desktop and a bottom sheet on mobile.

**Footer.** Dataset attribution (`meta.dataset`), the split windows from `meta.windows` ("Stage 1 trained on 1-28, calibrated on 29-34; final Stage 2 trained on 1-34; evaluated once on 35-49"), and a link to the GitHub repository (leave a placeholder constant at the top of `app.js`).

## 6. Honesty panel (required)

A "How to read this" panel, also shown as a short banner on first load (dismissible). It lists every string in `honesty` verbatim, plus the thresholds `d`, `d2`, `t_clear` and the validation-versus-test escalation rates (`kpis.validation_escalation_rate` vs `kpis.escalation_rate`). The word "simulated" must appear next to every cost figure, and the evaluation window "steps 35-49" next to every quality figure.

## 7. Things the app must NOT do

- Do not compute or show any metric that is not in the JSON or derivable with the formulas above.
- Do not describe it as real-time, live, production or "AI that stops fraud". It is a research evaluation on historical data.
- Do not show or imply real-world identities for transactions.
- Do not claim the cascade is a better model than Stage 2; it matches most of its recall at a fraction of the simulated cost.
- Do not draw null values as zero.
- Do not add analytics, trackers, or any network request other than the one `fetch` and the Chart.js script.

## 8. Acceptance checks (the user will verify these numbers by eye)

| What | Must show |
|---|---|
| "All" KPIs | 16,670 transactions; 15,538 cleared; 1,132 escalated (6.8%); 780 of 1,083 illicit caught |
| Recall kept vs graph-always ("All") | 780 / 795 = 98.1% |
| Savings at r = 5 / 20 / 100 ("All") | 77.7% / 88.8% / 92.3% |
| Step 35 batch | 1,341 transactions; 195 escalated; 175 of 182 illicit caught |
| Systems table, cascade row | precision 0.894, recall 0.720, F1 0.798, retention 0.981 |
| Per-step chart | all four lines collapse to about 0 from step 43 |
| Queue | no row has a p2 when route is Cleared |
| No data / failed fetch | readable error message, no blank page |

## 9. How to build it (send these as separate prompts)

1. "Read this spec. Build `index.html`, `styles.css` and `app.js` with the landing page, data loading with the error message, the batch picker and the KPI cards. Show the acceptance numbers for 'All' and for step 35."
2. "Add the simulated-cost card with the slider and the systems table."
3. "Add the two charts and the seed-spread panel, exactly as specified."
4. "Add the escalation queue, filter chips and the 'How this was decided' panel."
5. "Add the honesty panel and banner, the footer, the theme toggle, mobile layout and the accessibility checks. Then list anything in the spec you did not implement."

After each prompt, run `python -m http.server` inside the `web` folder, open http://localhost:8000 and check the table in section 8.

## 10. Deployment note (Day 11)

GitHub Pages can publish only the repository root or a `/docs` folder from a branch. To publish `web/`, use a small GitHub Actions workflow that uploads the `web` folder as the Pages artifact (`actions/upload-pages-artifact` with `path: web`, then `actions/deploy-pages`). Keep every path in the app relative.