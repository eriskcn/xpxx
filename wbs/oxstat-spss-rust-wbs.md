# Open-Source SPSS Alternative in Rust — Plan & WBS

## Project Vision

Statistical analysis suite matching core SPSS functionality. Rust for performance + safety. CLI-first, GUI later.

---

## Work Breakdown Structure

### Phase 1: Foundation (Months 1–3)

```
1.0 Core Infrastructure
├── 1.1 Project scaffolding (cargo workspace, CI, linting, docs)
├── 1.2 Data engine
│   ├── 1.2.1 Columnar in-memory data frame (Apache Arrow backing)
│   ├── 1.2.2 Missing value system (SYSMIS, user-defined missing)
│   ├── 1.2.3 Variable metadata (labels, value labels, measurement level)
│   └── 1.2.4 Variable types (numeric, string, date/time)
├── 1.3 File I/O
│   ├── 1.3.1 SPSS .sav reader/writer (binary format)
│   ├── 1.3.2 CSV import/export
│   ├── 1.3.3 Excel import (calamine crate)
│   └── 1.3.4 Parquet import/export
├── 1.4 Expression engine
│   ├── 1.4.1 COMPUTE-style expression parser
│   ├── 1.4.2 Built-in functions (math, string, date, statistical)
│   └── 1.4.3 Conditional logic (IF/RECODE/DO IF)
└── 1.5 Data transformations
    ├── 1.5.1 SELECT IF / FILTER
    ├── 1.5.2 SORT CASES
    ├── 1.5.3 AGGREGATE
    ├── 1.5.4 MERGE FILES (match files, add files)
    └── 1.5.5 RESHAPE (cases-to-vars, vars-to-cases)
```

### Phase 2: Descriptive Statistics (Months 3–5)

```
2.0 Statistical Procedures — Descriptives
├── 2.1 FREQUENCIES (freq tables, histograms, percentiles)
├── 2.2 DESCRIPTIVES (mean, sd, skew, kurtosis, z-scores)
├── 2.3 EXPLORE (stem-and-leaf, boxplots, normality tests)
├── 2.4 CROSSTABS (chi-square, phi, Cramér's V, lambda, odds ratio)
├── 2.5 MEANS / EXAMINE (subgroup statistics)
└── 2.6 Output system
    ├── 2.6.1 Structured output model (tables, charts, text, notes)
    ├── 2.6.2 Plain text renderer
    ├── 2.6.3 HTML renderer
    └── 2.6.4 PDF renderer (via printpdf or typst)
```

### Phase 3: Inferential Statistics (Months 5–8)

```
3.0 Statistical Procedures — Inferential
├── 3.1 T-TEST (one-sample, independent, paired, Levene's)
├── 3.2 ONEWAY (ANOVA, post-hoc: Tukey, Bonferroni, Scheffé, Games-Howell)
├── 3.3 GLM Univariate (factorial ANOVA, ANCOVA, eta-squared, partial eta²)
├── 3.4 CORRELATIONS (Pearson, Spearman, Kendall, partial)
├── 3.5 REGRESSION
│   ├── 3.5.1 Linear (enter, stepwise, diagnostics, collinearity)
│   ├── 3.5.2 Logistic (binary, multinomial, ordinal)
│   └── 3.5.3 Model diagnostics (residuals, Cook's D, VIF, DW)
├── 3.6 NONPAR TESTS
│   ├── 3.6.1 Mann-Whitney U
│   ├── 3.6.2 Wilcoxon signed-rank
│   ├── 3.6.3 Kruskal-Wallis
│   ├── 3.6.4 Friedman
│   └── 3.6.5 Kolmogorov-Smirnov, Shapiro-Wilk
└── 3.7 RELIABILITY (Cronbach's alpha, split-half, item-total)
```

### Phase 4: Advanced Methods (Months 8–12)

```
4.0 Advanced Statistical Procedures
├── 4.1 FACTOR (PCA, PAF, ML extraction; Varimax, Oblimin rotation; scree)
├── 4.2 CLUSTER
│   ├── 4.2.1 K-Means
│   ├── 4.2.2 Hierarchical (Ward, complete, average linkage; dendrogram)
│   └── 4.2.3 Two-Step
├── 4.3 DISCRIMINANT (linear, classification tables, Wilks' lambda)
├── 4.4 SURVIVAL
│   ├── 4.4.1 Kaplan-Meier
│   ├── 4.4.2 Cox regression
│   └── 4.4.3 Life tables
├── 4.5 MIXED (linear mixed models, random effects, ICC)
├── 4.6 GLM Repeated Measures (Mauchly's, Greenhouse-Geisser, Huynh-Feldt)
└── 4.7 Weighted analysis (WEIGHT BY support across all procedures)
```

### Phase 5: Syntax & CLI (Months 4–6, parallel)

```
5.0 SPSS Syntax Compatibility Layer
├── 5.1 Syntax parser (pest or winnow PEG grammar)
│   ├── 5.1.1 Core syntax (commands, subcommands, keywords, slashes)
│   ├── 5.1.2 String/numeric literals, variable lists, TO/THRU
│   └── 5.1.3 Macro facility (DEFINE/!ENDDEFINE, basic)
├── 5.2 Command dispatcher
├── 5.3 CLI interface
│   ├── 5.3.1 Batch mode (run .sps files)
│   ├── 5.3.2 Interactive REPL
│   └── 5.3.3 Pipe-friendly (stdin/stdout)
└── 5.4 Error reporting (line numbers, suggestions, SPSS-like messages)
```

### Phase 6: GUI (Months 10–14)

```
6.0 GUI Application
├── 6.1 Data view (spreadsheet-like, Variable View / Data View tabs)
├── 6.2 Output viewer (rendered tables + charts, scrollable)
├── 6.3 Syntax editor (highlighting, autocomplete, run selection)
├── 6.4 Dialog boxes for procedures (point-and-click → generates syntax)
├── 6.5 Chart builder
│   ├── 6.5.1 Bar, line, scatter, histogram, boxplot, pie
│   ├── 6.5.2 Interactive (zoom, tooltip)
│   └── 6.5.3 Export (PNG, SVG, PDF)
└── 6.6 Framework: egui or Tauri (web-based GUI, Rust backend)
```

### Phase 7: Ecosystem (Months 12–16)

```
7.0 Ecosystem & Integration
├── 7.1 Python bindings (PyO3)
├── 7.2 R bindings (extendr)
├── 7.3 WASM build (run in browser)
├── 7.4 Plugin system (user-defined procedures in Rust or Python)
├── 7.5 Documentation site (mdBook)
├── 7.6 SPSS syntax migration tool (parse .sps → compatibility report)
└── 7.7 Sample datasets + tutorials
```

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| In-memory format | Apache Arrow (arrow-rs) | Zero-copy, columnar, interop with Parquet/IPC |
| Missing values | Enum wrapper per cell | SPSS has SYSMIS + up to 3 user-missing per var |
| Syntax parser | PEG (pest/winnow) | SPSS syntax is context-sensitive but PEG handles it |
| Numeric backend | nalgebra + faer | faer for LAPACK-class linear algebra, nalgebra for convenience |
| GUI | Tauri v2 | Rust backend, web frontend, cross-platform, accessible |
| Concurrency | rayon for data ops | Embarrassingly parallel row/column operations |

---

## Crate Workspace Layout

```
oxstat/                        # working name
├── Cargo.toml                 # workspace root
├── crates/
│   ├── oxstat-core/           # data frame, missing values, metadata
│   ├── oxstat-io/             # .sav, CSV, Excel, Parquet readers/writers
│   ├── oxstat-expr/           # expression parser + evaluator
│   ├── oxstat-transform/      # data transformations
│   ├── oxstat-stats/          # all statistical procedures
│   ├── oxstat-syntax/         # SPSS syntax parser + command dispatch
│   ├── oxstat-output/         # output model + renderers
│   ├── oxstat-chart/          # chart generation
│   └── oxstat-cli/            # CLI binary
├── gui/                       # Tauri app
├── bindings/
│   ├── python/                # PyO3
│   └── r/                     # extendr
└── tests/
    ├── reference/             # SPSS output files for comparison
    └── integration/           # end-to-end syntax → output tests
```

---

## Milestones & Success Criteria

| Milestone | Target | Verification |
|---|---|---|
| M1: Read .sav, show DESCRIPTIVES | Month 3 | Load real SPSS file, output matches SPSS |
| M2: Core descriptives + crosstabs | Month 5 | 10 common procedures produce correct output |
| M3: Regression + ANOVA | Month 8 | Results match SPSS to 3 decimal places |
| M4: Syntax compatibility | Month 8 | Run 50 real-world .sps files without error |
| M5: GUI MVP | Month 12 | Data view + run syntax + view output |
| M6: Public beta | Month 14 | Docs, installers, 80%+ of common SPSS workflows |

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| .sav format underdocumented | High | Use GNU PSPP source as reference, test against real files |
| Numerical accuracy vs SPSS | High | Test against SPSS output + NIST StRD reference datasets |
| SPSS syntax is huge | Med | Prioritize top-30 commands by usage frequency, not completeness |
| GUI complexity | Med | CLI-first; GUI generates syntax, doesn't bypass it |
| Scope creep | Med | Each procedure = separate PR, tested independently |

---

## Prior Art to Study

- **GNU PSPP** — C, GPL. Most complete open SPSS clone. Study syntax parser + .sav format handling.
- **Jamovi** — R-based GUI. Study UX decisions.
- **JASP** — Bayesian focus. Study output presentation.
- **Polars** — Rust data frame. Study Arrow integration patterns.
