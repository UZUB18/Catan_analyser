# UI Improvements Backlog (Catan Analyzer)

This file lists practical UI upgrades to improve usability, clarity, speed-perception, and trust.

---

## 1) Analysis Feedback & Responsiveness

### 1.1 Live progress bar + ETA
- Show progress for long-running modes (`phase_rollout_mc`, `mcts_lite_opening`, `hybrid_opening`).
- Display:
  - current stage (e.g., "Running MCTS-lite…"),
  - percentage,
  - elapsed + estimated time remaining.

### 1.2 Cancel analysis button
- Let users stop an in-progress run without closing the app.
- Keep last successful results visible.

### 1.3 "Fast Preview" pass
- Run a quick low-budget estimate first, then refine in background.
- Users get immediate rough guidance.

---

## 2) Parameter UX (Controls Panel)

### 2.1 Preset chips by goal
- Buttons like: `Fast`, `Balanced`, `Tournament`, `Deep`.
- One-click parameter population, currently aligned with tuning logic.

### 2.2 Inline tooltips for every parameter
- Explain each input in plain language.
- Add recommended ranges and performance impact.

### 2.3 Constraint warnings before analyze
- Highlight invalid or risky combinations (e.g., huge iterations with low workers).
- Show expected runtime tier: `~1s`, `~5s`, `~20s+`.

---

## 3) Board Interaction Upgrades

### 3.1 Hover inspection
- Hover a vertex to preview:
  - adjacent resources,
  - pip total,
  - port info,
  - rank snapshot.

### 3.2 Click modes (selector toolbar)
- `Inspect`
- `Knowledge Test Pick`
- `Compare Two Vertices`
- Prevent ambiguous click behavior.

### 3.3 Stronger highlight language
- Different ring styles for:
  - engine top picks,
  - draft sequence picks,
  - user test picks,
  - selected vertex.

---

## 4) Results Table & Explainability

### 4.1 Sortable columns
- Allow sorting by any metric (score, yield, risk, tempo, etc.).

### 4.2 Pinned comparison tray
- Pin up to 3 vertices and compare metrics side-by-side.

### 4.3 Explainability "Why this rank?" drawer
- For selected vertex, show top feature contributions and penalties.

### 4.4 Sensitivity badges
- Indicate whether ranking is stable under small parameter changes.

---

## 5) Knowledge Test UX

### 5.1 Dedicated test panel
- Show:
  - selected picks,
  - current completeness (0/4…4/4),
  - score breakdown.

### 5.2 Round history
- Keep recent attempt history with board seed and score.

### 5.3 Shareable challenge mode
- Export/import board seed + mode + config for friends.

---

## 6) Visual Design & Readability

### 6.1 Theme switcher
- Light / dark / high-contrast themes.

### 6.2 Improved typography hierarchy
- Better distinction between section titles, labels, and data.

### 6.3 Color-safe palette
- Ensure overlays remain distinct for color-vision deficiencies.

### 6.4 Optional animation controls
- Toggle reduced motion for users who prefer minimal animation.

---

## 7) Layout & Navigation

### 7.1 Resizable split panes
- Let users resize board vs right-side panel live.

### 7.2 Collapsible sections
- Collapse advanced parameter blocks to reduce clutter.

### 7.3 Sticky summary header
- Keep key outputs visible while scrolling details.

---

## 8) Accessibility

### 8.1 Keyboard navigation
- Tab through controls and table.
- Arrow navigation for vertex rows.

### 8.2 Screen-reader friendly labels
- Improve naming for controls and result fields.

### 8.3 Larger UI scale option
- Built-in UI zoom presets for readability.

---

## 9) Reliability & Trust UX

### 9.1 Reproducibility panel
- Always show seed, mode, and effective config used for current result.

### 9.2 Confidence indicators
- Add uncertainty ribbons for stochastic modes.

### 9.3 Mode comparison snapshot
- Compare top-8 overlap across selected modes.

---

## Suggested implementation order

1. Progress bar + cancel + runtime estimate (highest UX impact)  
2. Tooltips + sortable table + pinned comparison  
3. Hover inspect + click mode toolbar  
4. Accessibility/theme enhancements  
5. Challenge/share and advanced explainability

