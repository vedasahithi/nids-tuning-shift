"""Build the tables of the paper from the stored results (no number is typed by hand).

Usage (from the repository root):
    python scripts/make_paper_tables.py

Writes paper/tables/<id>.md and <id>.csv for:
    T1_search     Table 1  search ranges and budgets (transcribed from configs/)
    T2_detectors  Table 2  detectors on the official split (T2_detectors_full.csv also has the test accuracy)
    T3_contrasts  Table 3  paired contrasts H1-H10 on test macro-F1 (from results/stats_report)
    T4_split      Table 4  decomposition of the gain of a random split
    T5_perclass   Table 5  test support and F1 by class (from results/per_class_eval)
    S1_shift      Table S1 shift between the partitions, label conflicts, bounds, label agreement
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

R = ROOT / "results"
OUT = ROOT / "paper" / "tables"
FAMILY = {"logreg": "linear", "rf": "tree ensemble", "xgb": "gradient boosting", "lgbm": "gradient boosting",
          "hgb": "gradient boosting", "hgb_default": "gradient boosting", "hgb_hsboa": "gradient boosting",
          "hgb_sboa": "gradient boosting", "hgb_random": "gradient boosting", "mlp": "neural",
          "cnn_bilstm": "neural", "qrnn_untuned": "neural (QRNN)", "qrnn_hsboa": "neural (QRNN)",
          "qrnn_random": "neural (QRNN)", "qrnn_random_d20": "neural (QRNN)"}   # used by make_figures.py
LABEL = {"lgbm_default": "LightGBM default", "logreg": "Logistic reg.", "rf": "RF+RS", "xgb": "XGBoost+RS",
         "lgbm": "LightGBM+RS", "hgb": "HGB+RS (HP)", "hgb_default": "HGB default", "hgb_hsboa": "HGB+HSBOA",
         "hgb_sboa": "HGB+SBOA", "hgb_random": "HGB+RS", "mlp": "MLP", "cnn_bilstm": "CNN-BiLSTM",
         "qrnn_untuned": "QRNN plain", "qrnn_hsboa": "QRNN+HSBOA†", "qrnn_random": "QRNN+RS",
         "qrnn_random_d20": "QRNN+RS d20"}
MINUS = "−"


def red(text: str) -> str:
    """Revision marker: [[...]] is shown in red by the manuscript builder (set REVISION_MARK=1)."""
    return f"[[{text}]]" if os.environ.get("REVISION_MARK") else text


def ms(x, d=3, sep=" ± ") -> str:
    x = pd.Series(x).dropna()
    return f"{x.mean():.{d}f}{sep}{x.std():.{d}f}" if len(x) else "–"


def m(x, d=3) -> str:
    x = pd.Series(x).dropna()
    return f"{x.mean():.{d}f}" if len(x) else "–"


def signed(x: float, d: int = 3) -> str:
    return f"{x:+.{d}f}".replace("-", MINUS)


def save(name: str, df: pd.DataFrame, caption: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    cols = [str(c) for c in df.columns]
    lines = [f"**{caption}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {len(df)} rows x {len(cols)} cols")


def results(folder: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((R / folder).glob("*/seed_*/result.json"))]


_META = {"val_macro_f1", "fitness", "wall_s", "gradient_steps", "trial", "cached", "val_macro_f1_weighted"}


def distinct_trainings(folder: str, name: str, seed: int) -> int:
    """Distinct settings in the full search history (trials.csv), counting trials reused after an interrupted
    attempt; result.json's distinct_trials counts only the last attempt, so it is not used here."""
    p = R / folder / name / f"seed_{seed:02d}" / "trials.csv"
    if not p.exists():
        return 0
    t = pd.read_csv(p)
    return int(len(t[[c for c in t.columns if c not in _META]].round(9).drop_duplicates()))


def detector_rows() -> pd.DataFrame:
    rows = []
    for r in results("e2_unsw_baselines"):
        rows.append({"model": r["model"], "seed": r["seed"], "val": r["val"]["macro_f1"], "test": r["test"]["macro_f1"],
                     "acc": r["test"]["accuracy"], "wall": r["wall_s"] / 60,
                     "evals": distinct_trainings("e2_unsw_baselines", r["model"], r["seed"])})
    acc = pd.read_csv(R / "budget_audit" / "accounting.csv").set_index(["arm", "seed"])["distinct_settings"]
    for r in results("e1b_unsw_gbdt"):
        ev = 0 if r["tuner"] == "default" else int(acc.loc[(f"hgb_{r['tuner']}", r["seed"])])
        rows.append({"model": "hgb_default" if r["tuner"] == "default" else f"hgb_{r['tuner']}", "seed": r["seed"],
                     "val": r["val"]["macro_f1"], "test": r["test"]["macro_f1"], "acc": r["test"]["accuracy"],
                     "wall": r["wall_s"] / 60, "evals": ev})
    for r in results("e1_unsw_tuners"):
        rows.append({"model": f"qrnn_{r['tuner']}", "seed": r["seed"], "val": r["val"]["macro_f1"],
                     "test": r["test"]["macro_f1"], "acc": r["test"]["accuracy"], "wall": r["wall_s"] / 60,
                     "evals": distinct_trainings("e1_unsw_tuners", r["tuner"], r["seed"])})
    # untuned LightGBM: the default arm of E6 (one fit, no search)
    e6 = pd.read_csv(R / "e6_unsw_shift" / "runs.csv")
    for r in e6[e6.arm == "default"].itertuples(index=False):
        rows.append({"model": "lgbm_default", "seed": r.seed, "val": r.val_macro_f1, "test": r.test_macro_f1,
                     "acc": r.test_accuracy, "wall": r.wall_min, "evals": 0})
    return pd.DataFrame(rows).drop_duplicates(["model", "seed"], keep="last")


def t_detectors() -> None:
    df = detector_rows()
    out = []
    for k, g in df.groupby("model"):
        w = g.wall.mean()
        out.append({"Model": LABEL[k], "Val. F1": m(g.val), "Test F1": ms(g.test, sep="±"), "Test acc.": m(g.acc),
                    "Evals": f"{int(g.evals.mean() + 0.5)}", "Min": f"{w:.0f}" if w >= 1 else "<1",
                    "_k": g.test.mean()})
    t = pd.DataFrame(out).sort_values("_k", ascending=False).drop(columns="_k")
    t.to_csv(OUT / "T2_detectors_full.csv", index=False, encoding="utf-8")      # with the test accuracy
    lg = df[df.model == "lgbm_default"].test.mean()
    lr = df[df.model == "lgbm"].test.mean()
    save("T2_detectors", t.drop(columns="Test acc."),
         "Table 2. Detectors on the official split (10 classes; mean of 10 seeds, test ± sd). RS = random search; "
         "Evals = distinct trainings per run (0 = no search; submissions are in Table 1); Min = minutes per run on one CPU thread; HP = "
         "hyper-parameters only. The two default rows are not searched: HGB default (20 features, α = 0.5) follows a validation-only "
         + red("18-cell pilot on three seeds; ") + "LightGBM default uses library settings with the same features and "
         f"α and is the model of Table 4 and Fig. 1 (tuned {lr:.3f}, default {lg:.3f}). † In QRNN+HSBOA the harmonic term "
         "never starts (three iterations).")


def t_contrasts() -> None:
    """H1-H10 on test macro-F1: mean paired difference, 95% bootstrap interval, Holm-adjusted two-sided p."""
    c = pd.read_csv(R / "stats_report" / "contrasts.csv")
    c = c[c.metric == "test_macro_f1"].set_index("b")
    e6 = pd.read_csv(R / "stats_report" / "contrasts_e6.csv").set_index("hypothesis")

    def p(x: float) -> str:
        return f"{x:.2f}" if x < 0.995 and x >= 0.1 else (f"{x:.3f}" if x < 0.1 else "1")

    nb = " "        # keeps "(low, high)" on one line

    def cell(d, lo, hi) -> str:
        return f"{signed(d)} ({signed(lo)},{nb}{signed(hi)})"

    h1 = c[c.hypothesis == "H1"]
    names = {"qrnn_hsboa": "QRNN+HSBOA", "qrnn_random": "QRNN+RS", "qrnn_random_d20": "QRNN+RS d20",
             "cnn_bilstm": "CNN-BiLSTM", "mlp": "MLP"}
    rows = []
    for r in h1.sort_values("mean_diff", ascending=False).itertuples():
        rows.append((f"H1 HSBOA {MINUS} {names[r.Index]}", cell(r.mean_diff, r.ci95_low, r.ci95_high),
                     p(r.p_holm_h1_h4_by_metric_8)))
    for h, b, lab in (("H2", "hgb_random", "RS"), ("H3", "hgb_sboa", "SBOA"), ("H4", "hgb_default", "default")):
        r = c.loc[b]
        rows.append((f"{h} HSBOA {MINUS} {lab}", cell(r.mean_diff, r.ci95_low, r.ci95_high),
                     p(r.p_holm_h1_h4_by_metric_8)))
    labs = {"H5": "standard", "H6": "RS", "H7": "SBOA", "H8": "PSO", "H9": "shift", "H10": "default"}
    for h, lab in labs.items():
        r = e6.loc[h]
        first = "w.t." if h == "H9" else "shift"
        rows.append((f"{h} {first} {MINUS} {lab}", cell(r.mean_diff, r.ci95_low, r.ci95_high),
                     p(r.p_holm_two_sided)))
    save("T3_contrasts", pd.DataFrame(rows, columns=["Contrast", "Difference (95% CI)", "p (Holm)"]),
         "Table 3. Paired contrasts on test macro-F1 (mean difference over 10 seeds, 95% bootstrap interval, "
         "Holm-adjusted two-sided Wilcoxon p within H1-H4 (8 comparisons) and H5-H10 (6)); only H1 is supported. "
         "HSBOA = HGB tuned by HSBOA (H1 is " + red("five separate contrasts with the tuned neural models") + "); shift = LightGBM tuned by HSBOA with the "
         "shift-aware fitness (standard = standard fitness; w.t. = weighted training); RS = random search.")


def t_search() -> None:
    """Search spaces and budgets, transcribed from configs/experiments/*.yaml and configs/model.yaml."""
    lgbm = ("rate 10^[-2.5,-0.5]; trees 100-400; leaves 15-127; min child 5-100; L2 10^[-6,1]; "
            "rows 0.5-1; columns 0.3-1")
    rows = [
        ("LightGBM + mask", "mask; " + lgbm, "400; RS 373"),
        ("HGB + mask", "mask; rate 10^[-2.5,-0.5]; iterations 100-400; leaves 15-127; min leaf 5-100; "
                       "L2 10^[-6,0]", "400; RS 373"),
        ("HGB, ablation", "no mask (top 20 features); rate 10^[-2.5,-0.5]; iterations 100-400; leaves 15-127; "
                          "min leaf 5-100; L2 10^[-6,0]; default settings as first candidate", "60 (pop. 10)"),
        ("MLP", "layers 1-3; units 2^[4,8]; dropout 0-0.5; rate 10^[-4,-2]; L2 10^[-7,-3]; batch 2^[6,9]",
         "39"),
        ("CNN-BiLSTM", "filters 2^[4,7]; kernel 2-5; LSTM units 2^[4,7]; dropout 0-0.5; rate 10^[-4,-2]; "
                       "batch 2^[6,9]", "39"),
        ("QRNN", "rate 10^[-4,-1.5]; units 2^[3,6]; window 1-4; dropout 0-0.5; L2 10^[-7,-3]; batch 2^[6,9]"
                 , "48; RS 39"),
        ("XGBoost", "rate 10^[-2.5,-0.5]; trees 100-400; depth 3-12; min weight 10^[-1,1.5]; L2 10^[-6,1]; "
                    "rows 0.5-1; columns 0.3-1", "32"),
        ("Random forest", "trees 100-400; depth 6-40; min leaf 1-10; features 0.1-1", "32"),
        ("Logistic reg.", "C 10^[-3,3] (7 values); α 0, 0.5, 1", "21"),
    ]
    save("T1_search", pd.DataFrame(rows, columns=["Model", "Search ranges", red("Submitted")]),
         "Table 1. Search ranges and " + red("submitted candidates") + " per run (RS = random search; PSO 400; " + red("distinct trainings, 372-375 for the 400-submission arms, are in Table 2 and Section 4.3") + "). 10^[a,b] and "
         "2^[a,b] are uniform in the exponent; integers are rounded; every model also searches the class-weight power α in [0, 1]. mask = 42 "
         "coordinates in [0, 1], on if above 0.5, the most informative columns added until three are on. "
         "Without the mask, LightGBM, XGBoost, random forest and HGB use random search (32 trainings). Neural "
         "models: early stopping on validation loss (patience 5, final 10), at most 100 epochs per trial, "
         "200 final.")


def t_shift() -> None:
    a = pd.read_csv(R / "diag_shift" / "domain_auc.csv").groupby("comparison")["auc"].mean()
    d = json.loads((R / "diag_shift" / "duplicates.json").read_text(encoding="utf-8"))
    bnd = pd.read_csv(R / "ceiling_bounds" / "bounds.csv").set_index("set").loc["test partition"]
    lac = pd.read_csv(R / "label_agreement" / "summary.csv").set_index("quantity")
    la = lac["value"]
    ex = lac.loc["excess disagreement: cross minus standardised within-train"]
    rows = [
        ("Domain AUC: train vs test partition", f"{a['train_vs_test']:.3f}"),
        ("Domain AUC: class mix matched", f"{a['matched_vs_test']:.3f}"),
        ("… with a copy of another class (all train records)", f"{100 * d['conflict_row_share']:.1f}%"),
        ("Test macro-F1: majority reference", f"{bnd.majority:.3f}"),
        ("Test macro-F1: best assignment found", f"{bnd.achievable:.3f}"),
        ("Test macro-F1: upper bound, any assignment", f"{bnd.upper:.3f}"),
        ("Twin has another label: test vs train / within train",
         f"{la['cross-partition pairwise disagreement (test vs train twin)']:.3f} / "
         f"{la['within-train disagreement standardised to matched test class mix']:.3f}"),
        ("Excess disagreement (95% CI)", f"{ex.value:.3f} ({ex.ci_lo:.3f}-{ex.ci_hi:.3f})"),
    ]
    save("S1_shift", pd.DataFrame(rows, columns=["Quantity", "Value"]),
         "Table S1. Shift between the partitions (domain classifier on features only, 3 seeds), label conflicts, "
         "macro-F1 bounds for classifiers of the 42 features, and label agreement of test records with identical "
         "training records (bootstrap over groups).")


def t_split() -> None:
    p = pd.read_csv(R / "split_control" / "protocol.csv")
    st = pd.read_csv(R / "split_control" / "split_stats.csv").groupby("split").mean(numeric_only=True)
    f = p[p.settings == "default"].groupby("split")["macro_f1"]
    desc = {"official": "Official", "R": "R: random", "G": "G: no shared", "P": "P: no shared, class counts"}
    rows = []
    for k in ("official", "R", "G", "P"):
        rows.append({"Split": desc[k], "Twin": f"{100 * st.loc[k, 'test_rows_with_train_twin']:.1f}%",
                     "Macro-F1": ms(f.get_group(k), 4)})
    dec = pd.read_csv(R / "split_control" / "decomposition.csv")
    dec = dec[dec.settings == "default"].set_index("component")
    lab = {"total_gain_R_minus_official": "Gain R − official",
           "duplicate_exposure_R_minus_G": "Duplicates R − G",
           "class_mix_G_minus_P": "Class mix G − P",
           "residual_shift_P_minus_official": "Shift P − official"}
    for k, v in lab.items():
        r = dec.loc[k]
        rows.append({"Split": v, "Twin": "", "Macro-F1": f"{r['mean']:+.4f} ({r.ci95_low:+.4f}, {r.ci95_high:+.4f})"})
    save("T4_split", pd.DataFrame(rows),
         "Table 4. Decomposition of the gain of a random split (default LightGBM, 10 seeds; mean ± sd, "
         "differences with 95% bootstrap intervals). Mean sizes (train / test): R and official 175,341 / 82,332; "
         "G 175,379 / 82,294; P 175,340 / 78,777 (P keeps the official training counts and has a test side about "
         "4% smaller in every class). Twin = share of test records with an identical training record.")


def t_perclass() -> None:
    """Support and per-class F1 on the official test partition (results/per_class_eval)."""
    d = pd.read_csv(R / "per_class_eval" / "per_class.csv")
    arms = [("lgbm_default", "LightGBM default"), ("hgb_hsboa", "HGB+HSBOA"), ("mlp", "MLP")]
    classes = list(d[d.arm == "lgbm_default"]["class"])
    rows = []
    for c in classes:
        sup = int(d[(d.arm == "lgbm_default") & (d["class"] == c)].support.iloc[0])
        row = {"Class": "Recon." if c == "Reconnaissance" else c, "n": str(sup)}
        for a, lab in arms:
            r = d[(d.arm == a) & (d["class"] == c)].iloc[0]
            half = max(r.f1_seed_hi - r.f1_seed_lo, r.f1_grouped_hi - r.f1_grouped_lo) / 2
            row[lab] = f"{r.f1:.3f} ±{half:.3f}" if a == "lgbm_default" else f"{r.f1:.3f}"
        rows.append(row)
    save("T5_perclass", pd.DataFrame(rows),
         "Table 5. Test support (n) and F1 by class on the official partition (mean of 10 seeds). ± = half-width "
         "of the wider 95% interval (bootstrap over seeds or over groups of identical test records, 2,000 "
         "resamples) for LightGBM default; the intervals of all arms are in the repository. Final models refitted "
         "from the stored settings; CNN-BiLSTM omitted (refit not exact). Recon. = Reconnaissance.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.*"):
        f.unlink()
    for fn in (t_search, t_detectors, t_contrasts, t_shift, t_split, t_perclass):
        fn()
