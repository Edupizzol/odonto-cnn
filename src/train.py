"""Fine-tuning de ResNet50 / EfficientNet-B0 (3 saídas multi-label).

Uso:
  python src/train.py --archs resnet50 efficientnet_b0 --seeds 42 43 44
Cada rodada grava em <out>/<arch>_s<seed>_pw<pw>/: best.pt, metrics.csv,
last.pt (retomada; removido ao final) e done.json (marca rodada concluída).
O conjunto de teste NÃO é tocado aqui.
"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from dataset import CLASSES, get_loaders, pos_weight
from models import ARCHS, build_model, param_groups


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True  # mais rápido; rodadas não são bit a bit idênticas


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # evita checkpoint corrompido se a sessão cair no meio da escrita


@torch.no_grad()
def predict(model, loader, crit, dev):
    model.eval()
    probs, ys, tot, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        out = out.float()
        tot += crit(out, y).item() * len(y); n += len(y)
        probs.append(torch.sigmoid(out).cpu()); ys.append(y.cpu())
    return tot / n, torch.cat(probs).numpy(), torch.cat(ys).numpy()


def metrics(probs, ys, thr=0.5):
    y, pred = ys.astype(int), (probs >= thr).astype(int)
    f1 = f1_score(y, pred, average=None, zero_division=0)
    rec = recall_score(y, pred, average=None, zero_division=0)
    auc = [roc_auc_score(y[:, i], probs[:, i]) for i in range(len(CLASSES))]
    m = {"f1_macro": f1.mean(), "auc_macro": float(np.mean(auc))}
    for i, c in enumerate(CLASSES):
        m[f"f1_{c}"], m[f"rec_{c}"], m[f"auc_{c}"] = f1[i], rec[i], auc[i]
    return {k: float(v) for k, v in m.items()}


def run(arch, seed, a):
    run_dir = os.path.join(a.out, f"{arch}_s{seed}_pw{a.pw}")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(f"{run_dir}/done.json"):
        print(f"[pulando] já concluído: {run_dir}"); return

    dev = "cuda"
    set_seed(seed)
    tr, va, _ = get_loaders(a.manifest, a.batch, a.workers, seed)
    pw = pos_weight(a.manifest)
    if a.pw == "sqrt":
        pw = pw.sqrt()
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw.to(dev))

    model = build_model(arch).to(dev)
    opt = torch.optim.AdamW(param_groups(model, arch, a.lr_backbone, a.lr_head),
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    start, best, best_ep, bad, hist = 0, -1.0, 0, 0, []
    last = f"{run_dir}/last.pt"
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, best_ep, bad, hist = ck["epoch"], ck["best"], ck["best_ep"], ck["bad"], ck["hist"]
        print(f"[retomando] {run_dir} na época {start + 1}")

    print(f"=== {arch} seed={seed} pw={a.pw} pos_weight={[round(v, 2) for v in pw.tolist()]} ===")
    for ep in range(start, a.epochs):
        torch.manual_seed(seed * 1000 + ep)
        model.train(); t0 = time.time(); run_loss, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss = crit(out.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_loss += loss.item() * len(y); n += len(y)
        sched.step()

        vloss, probs, ys = predict(model, va, crit, dev)
        m = metrics(probs, ys)
        hist.append({"epoch": ep + 1, "train_loss": run_loss / n, "val_loss": vloss,
                     "secs": time.time() - t0, **m})
        pd.DataFrame(hist).to_csv(f"{run_dir}/metrics.csv", index=False)

        if m["f1_macro"] > best:
            best, best_ep, bad = m["f1_macro"], ep + 1, 0
            save_atomic(model.state_dict(), f"{run_dir}/best.pt")
        else:
            bad += 1
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": ep + 1, "best": best, "best_ep": best_ep,
                     "bad": bad, "hist": hist}, last)

        print(f"ep {ep + 1:02d} | loss {run_loss / n:.3f} | val {vloss:.3f} | "
              f"F1m {m['f1_macro']:.3f} | AUCm {m['auc_macro']:.3f} | "
              f"rec_perip {m['rec_periapical']:.2f} | {time.time() - t0:.0f}s"
              f"{'  *' if best_ep == ep + 1 else ''}")
        if bad >= a.patience:
            print(f"early stopping (sem melhora há {a.patience} épocas)"); break

    json.dump({"best_f1_macro": best, "best_epoch": best_ep, "epochs_run": len(hist),
               "args": vars(a), "arch": arch, "seed": seed},
              open(f"{run_dir}/done.json", "w"), indent=1)
    if os.path.exists(last):
        os.remove(last)  # libera espaço no Drive
    print(f"concluído: melhor F1 macro {best:.3f} na época {best_ep}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--pw", choices=["sqrt", "full"], default="sqrt")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--manifest", default="/content/dentex_crops/manifest.csv")
    p.add_argument("--out", default="/content/drive/MyDrive/odonto-cnn/runs")
    a = p.parse_args()
    for arch in a.archs:
        for seed in a.seeds:
            run(arch, seed, a)