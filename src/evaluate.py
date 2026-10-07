"""Avaliação final no conjunto de TESTE. Rodar uma vez, depois de fechar o treino.

Para cada rodada: carrega best.pt, escolhe o limiar de cada classe maximizando o
F1 na VALIDAÇÃO e mede no TESTE. A incerteza vem de bootstrap por radiografia
(todos os dentes de uma radiografia entram ou saem juntos).
As previsões ficam em <results>/preds.npz, então reexecutar não precisa de GPU.
"""
import argparse, glob, json, os, re, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from dataset import CLASSES, TeethDataset
from models import build_model

warnings.filterwarnings("ignore", category=RuntimeWarning)
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
PERI = CLASSES.index("periapical")
BK = (["f1_macro", "auc_macro", "nll", "ece"]
      + [f"{k}_{c}" for c in CLASSES for k in ("f1", "rec", "spec", "auc")]
      + ["perip_rec_superior", "perip_rec_inferior", "perip_spec_superior",
         "perip_spec_inferior", "perip_rec_gap"])  # gap = inferior - superior


# ---------- métricas ----------
def auc_fast(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def safe_div(a, b):
    return np.where(b > 0, a / np.maximum(b, 1), np.nan)


def ece_binary(y, p, bins=15):
    ids = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = ids == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


def compute(y, p, thr, counts=False):
    pred = (p >= thr[None, :]).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(0); fp = ((pred == 1) & (y == 0)).sum(0)
    fn = ((pred == 0) & (y == 1)).sum(0); tn = ((pred == 0) & (y == 0)).sum(0)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    rec, spec = safe_div(tp, tp + fn), safe_div(tn, tn + fp)
    auc = np.array([auc_fast(y[:, c], p[:, c]) for c in range(y.shape[1])])
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    out = {"f1_macro": np.nanmean(f1), "auc_macro": np.nanmean(auc),
           "nll": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
           "ece": float(np.mean([ece_binary(y[:, c], p[:, c]) for c in range(y.shape[1])]))}
    for i, c in enumerate(CLASSES):
        out[f"f1_{c}"], out[f"rec_{c}"] = f1[i], rec[i]
        out[f"spec_{c}"], out[f"auc_{c}"] = spec[i], auc[i]
        if counts:
            out.update({f"tp_{c}": tp[i], f"fp_{c}": fp[i], f"fn_{c}": fn[i], f"tn_{c}": tn[i]})
    return out


def arcada_stats(yc, pred, mask):
    pos, neg = (yc == 1) & mask, (yc == 0) & mask
    rec = (pred & pos).sum() / pos.sum() if pos.sum() else np.nan
    spec = (~pred & neg).sum() / neg.sum() if neg.sum() else np.nan
    return rec, spec


def run_stats(yb, pb, thr, sup):
    m = compute(yb, pb, thr)
    pred = pb[:, PERI] >= thr[PERI]
    rs, ss = arcada_stats(yb[:, PERI], pred, sup)
    ri, si = arcada_stats(yb[:, PERI], pred, ~sup)
    m.update({"perip_rec_superior": rs, "perip_rec_inferior": ri,
              "perip_spec_superior": ss, "perip_spec_inferior": si, "perip_rec_gap": ri - rs})
    return m


def tune_thresholds(y, p):
    thr = []
    for c in range(y.shape[1]):
        yc, sc = y[:, c] == 1, []
        for t in GRID:
            pr = p[:, c] >= t
            tp, fp, fn = (pr & yc).sum(), (pr & ~yc).sum(), (~pr & yc).sum()
            sc.append(2 * tp / max(2 * tp + fp + fn, 1))
        sc = np.array(sc)
        cand = GRID[sc >= sc.max() - 1e-12]
        thr.append(float(cand[np.argmin(np.abs(cand - 0.5))]))  # empate: o mais próximo de 0,5
    return np.array(thr)


# ---------- previsões ----------
@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    return torch.cat([torch.sigmoid(model(x.to(dev)).float()).cpu() for x, _ in loader]).numpy()


def get_preds(runs, loaders, sizes, a):
    cache = os.path.join(a.results, "preds.npz")
    preds = {}
    if os.path.exists(cache) and not a.recompute:
        z = np.load(cache); preds = {k: z[k] for k in z.files}
    need = [r for r in runs if f"{r['name']}__val" not in preds or f"{r['name']}__test" not in preds]
    if need:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for r in need:
            m = build_model(r["arch"], pretrained=False).to(dev)
            m.load_state_dict(torch.load(f"{r['dir']}/best.pt", map_location=dev))
            for s in ("val", "test"):
                preds[f"{r['name']}__{s}"] = predict(m, loaders[s], dev)
            print("previsões:", r["name"])
        np.savez(cache, **preds)
    for r in runs:
        for s in ("val", "test"):
            assert preds[f"{r['name']}__{s}"].shape == (sizes[s], len(CLASSES)), \
                "cache incompatível com o manifest; rode com --recompute"
    return preds


# ---------- principal ----------
def main(a):
    os.makedirs(f"{a.results}/figs", exist_ok=True)
    sets = {s: TeethDataset(a.manifest, s, train=False) for s in ("val", "test")}
    loaders = {s: DataLoader(sets[s], batch_size=64, shuffle=False, num_workers=a.workers) for s in sets}
    y = {s: sets[s].labels.numpy().astype(int) for s in sets}
    sizes = {s: len(sets[s]) for s in sets}
    tdf, yt = sets["test"].df, y["test"]

    runs = []
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        m = re.match(r"(resnet50|efficientnet_b0)_s(\d+)_pw(.+)$", os.path.basename(d))
        if m and m.group(3) == a.pw and os.path.exists(f"{d}/done.json") and os.path.exists(f"{d}/best.pt"):
            runs.append({"name": os.path.basename(d), "arch": m.group(1), "seed": int(m.group(2)), "dir": d})
    if not runs:
        raise SystemExit("nenhuma rodada concluída encontrada em " + a.runs)
    archs = sorted({r["arch"] for r in runs})
    print("rodadas:", [r["name"] for r in runs])

    preds = get_preds(runs, loaders, sizes, a)

    # limiares (validação) e métricas pontuais (teste)
    thr_by, rows = {}, []
    for r in runs:
        pv, pt = preds[f"{r['name']}__val"], preds[f"{r['name']}__test"]
        thr = tune_thresholds(y["val"], pv); thr_by[r["name"]] = thr
        m = compute(yt, pt, thr, counts=True)
        rows.append({"run": r["name"], "arch": r["arch"], "seed": r["seed"],
                     **{f"thr_{c}": thr[i] for i, c in enumerate(CLASSES)}, **m,
                     "f1_macro_thr05": compute(yt, pt, np.full(len(CLASSES), 0.5))["f1_macro"]})
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(f"{a.results}/runs_test.csv", index=False)

    sup = (tdf.fdi.values // 10) <= 2  # FDI quadrantes 1 e 2 = arcada superior
    point = {r["name"]: run_stats(yt, preds[f"{r['name']}__test"], thr_by[r["name"]], sup) for r in runs}
    arr = {ar: np.array([[point[r["name"]][k] for k in BK] for r in runs if r["arch"] == ar]) for ar in archs}
    mean = {ar: np.nanmean(arr[ar], 0) for ar in archs}
    sd = {ar: np.nanstd(arr[ar], 0, ddof=1) for ar in archs}

    # bootstrap por radiografia
    groups = [np.where(tdf.image.values == u)[0] for u in pd.unique(tdf.image.values)]
    rng = np.random.default_rng(a.boot_seed)
    boot = {ar: np.full((a.B, len(BK)), np.nan) for ar in archs}
    for b in range(a.B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        acc = {ar: [] for ar in archs}
        for r in runs:
            m = run_stats(yt[idx], preds[f"{r['name']}__test"][idx], thr_by[r["name"]], sup[idx])
            acc[r["arch"]].append([m[k] for k in BK])
        for ar in archs:
            boot[ar][b] = np.nanmean(np.array(acc[ar]), 0)  # média entre seeds
        if (b + 1) % 500 == 0:
            print(f"bootstrap {b + 1}/{a.B}")

    # tabela média ± dp entre seeds + IC95% do bootstrap
    tab = []
    for ar in archs:
        lo, hi = np.nanpercentile(boot[ar], [2.5, 97.5], axis=0)
        for j, k in enumerate(BK):
            tab.append({"arch": ar, "metric": k, "mean_seeds": mean[ar][j], "sd_seeds": sd[ar][j],
                        "ci_lo": lo[j], "ci_hi": hi[j]})
    tab = pd.DataFrame(tab); tab.to_csv(f"{a.results}/summary_ci.csv", index=False)

    # comparação pareada (mesmas radiografias sorteadas nas duas redes)
    paired = None
    if {"resnet50", "efficientnet_b0"} <= set(archs):
        d = boot["resnet50"] - boot["efficientnet_b0"]
        pr = []
        for j, k in enumerate(BK):
            v = d[:, j][np.isfinite(d[:, j])]
            p = min(1.0, 2 * (min((v <= 0).sum(), (v >= 0).sum()) + 1) / (len(v) + 1))
            pr.append({"metric": k, "diff_resnet_minus_effnet": mean["resnet50"][j] - mean["efficientnet_b0"][j],
                       "ci_lo": np.percentile(v, 2.5), "ci_hi": np.percentile(v, 97.5), "p_boot": p})
        paired = pd.DataFrame(pr); paired.to_csv(f"{a.results}/paired_resnet_vs_effnet.csv", index=False)

    # lesão periapical por arcada
    ar_rows = []
    for ar in archs:
        lo, hi = np.nanpercentile(boot[ar], [2.5, 97.5], axis=0)
        for arc, mask in (("superior", sup), ("inferior", ~sup)):
            jr, js = BK.index(f"perip_rec_{arc}"), BK.index(f"perip_spec_{arc}")
            ar_rows.append({"arch": ar, "arcada": arc, "n_dentes": int(mask.sum()),
                            "n_periapical": int(yt[mask, PERI].sum()),
                            "rec_mean": mean[ar][jr], "rec_sd": sd[ar][jr], "rec_lo": lo[jr], "rec_hi": hi[jr],
                            "spec_mean": mean[ar][js], "spec_sd": sd[ar][js], "spec_lo": lo[js], "spec_hi": hi[js]})
    arc_df = pd.DataFrame(ar_rows); arc_df.to_csv(f"{a.results}/periapical_por_arcada.csv", index=False)

    # figuras
    fig, axs = plt.subplots(len(archs), len(CLASSES), figsize=(3.2 * len(CLASSES), 3 * len(archs)), squeeze=False)
    for i, ar in enumerate(archs):
        sub = runs_df[runs_df.arch == ar]
        for j, c in enumerate(CLASSES):
            cm = np.array([[sub[f"tn_{c}"].mean(), sub[f"fp_{c}"].mean()],
                           [sub[f"fn_{c}"].mean(), sub[f"tp_{c}"].mean()]])
            ax = axs[i][j]; ax.imshow(cm, cmap="Blues")
            for (u, v), val in np.ndenumerate(cm):
                ax.text(v, u, f"{val:.1f}", ha="center", va="center",
                        color="white" if val > cm.max() / 2 else "black")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["não", "sim"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["não", "sim"])
            ax.set_title(f"{ar}\n{c}", fontsize=9); ax.set_xlabel("previsto"); ax.set_ylabel("real")
    fig.tight_layout(); fig.savefig(f"{a.results}/figs/confusion.png", dpi=200); plt.close(fig)

    col = {"resnet50": "tab:blue", "efficientnet_b0": "tab:orange"}
    fig, axs = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 3.8))
    for j, c in enumerate(CLASSES):
        for r in runs:
            fpr, tpr, _ = roc_curve(yt[:, j], preds[f"{r['name']}__test"][:, j])
            axs[j].plot(fpr, tpr, color=col[r["arch"]], alpha=0.6, lw=1)
        axs[j].plot([0, 1], [0, 1], "k--", lw=0.8)
        axs[j].set_title(c); axs[j].set_xlabel("1 - especificidade"); axs[j].set_ylabel("sensibilidade")
    axs[-1].legend(handles=[Line2D([0], [0], color=col[ar], label=ar) for ar in archs], loc="lower right")
    fig.tight_layout(); fig.savefig(f"{a.results}/figs/roc.png", dpi=200); plt.close(fig)

    json.dump({"args": vars(a), "runs": [r["name"] for r in runs],
               "thresholds": {k: v.tolist() for k, v in thr_by.items()}},
              open(f"{a.results}/eval_config.json", "w"), indent=1)

    # saída no terminal
    show = ["f1_macro", "auc_macro", "nll", "ece"] + [f"{k}_{c}" for c in CLASSES for k in ("f1", "rec", "spec", "auc")]
    lines = []
    for k in show:
        row = {"métrica": k}
        for ar in archs:
            t = tab[(tab.arch == ar) & (tab.metric == k)].iloc[0]
            row[ar] = f"{t.mean_seeds:.3f} ± {t.sd_seeds:.3f} [{t.ci_lo:.3f}, {t.ci_hi:.3f}]"
        lines.append(row)
    print("\n=== TESTE: média ± dp entre seeds [IC95% bootstrap por radiografia] ===")
    print(pd.DataFrame(lines).to_string(index=False))
    if paired is not None:
        print("\n=== ResNet50 - EfficientNet-B0 (IC95% e p por bootstrap pareado) ===")
        print(paired.round(3).to_string(index=False))
    print("\n=== Lesão periapical por arcada ===")
    print(arc_df.round(3).to_string(index=False))
    print("\nlimiares por rodada:\n", runs_df[["run"] + [f"thr_{c}" for c in CLASSES]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="/content/drive/MyDrive/odonto-cnn/runs")
    p.add_argument("--results", default="/content/drive/MyDrive/odonto-cnn/results")
    p.add_argument("--manifest", default="/content/dentex_crops/manifest.csv")
    p.add_argument("--pw", default="sqrt")
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--recompute", action="store_true")
    main(p.parse_args())