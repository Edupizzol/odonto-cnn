"""Gera os recortes por dente do DENTEX e o manifest.csv com o split.

Roda no Colab depois de baixar o DENTEX em /content/dentex e o
validation_triple.json em /content/dentex_raw/DENTEX/.
Uso: python src/prepare_crops.py
"""
import json, os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT, OUT = "/content/dentex", "/content/dentex_crops"
SEED, SIZE, MARGIN = 42, 224, 0.15
CLASSES = ["caries", "deep_caries", "periapical"]
LABEL_MAP = {1: "caries", 3: "deep_caries", 2: "periapical"}  # category_id_3; 0 = impactado (descartado)
SOURCES = [
    (f"{ROOT}/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
     f"{ROOT}/training_data/quadrant-enumeration-disease/xrays"),
    ("/content/dentex_raw/DENTEX/validation_triple.json",
     f"{ROOT}/validation_data/quadrant_enumeration_disease/xrays"),
]


def collect_teeth():
    rows = []
    for json_path, img_dir in SOURCES:
        coco = json.load(open(json_path))
        info = {i["id"]: i for i in coco["images"]}
        groups = {}
        for a in coco["annotations"]:
            cls = LABEL_MAP.get(a["category_id_3"])
            if cls is None:
                continue
            key = (a["image_id"], a["category_id_1"], a["category_id_2"])
            g = groups.setdefault(key, {"labels": set(), "boxes": []})
            g["labels"].add(cls)
            g["boxes"].append(a["bbox"])
        for (img_id, q, n), g in groups.items():
            im, b = info[img_id], g["boxes"]
            rows.append(dict(
                image=im["file_name"], img_path=f"{img_dir}/{im['file_name']}",
                width=im["width"], height=im["height"], fdi=(q + 1) * 10 + (n + 1),
                x1=min(x for x, y, w, h in b), y1=min(y for x, y, w, h in b),
                x2=max(x + w for x, y, w, h in b), y2=max(y + h for x, y, w, h in b),
                **{c: int(c in g["labels"]) for c in CLASSES}))
    return pd.DataFrame(rows)


def assign_split(df):
    imgs = df.groupby("image")["periapical"].max().reset_index()
    tr, tmp = train_test_split(imgs, test_size=0.30, stratify=imgs["periapical"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["periapical"], random_state=SEED)
    m = {**{i: "train" for i in tr.image}, **{i: "val" for i in va.image}, **{i: "test" for i in te.image}}
    df["split"] = df["image"].map(m)
    return df


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def crop_tooth(img, r):
    # janela quadrada da radiografia original, centrada no dente (sem barras pretas)
    H, W = img.shape
    cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
    side = max(r.x2 - r.x1, r.y2 - r.y1) * (1 + 2 * MARGIN)
    side = int(min(side, H, W))
    x1 = int(min(max(cx - side / 2, 0), W - side))
    y1 = int(min(max(cy - side / 2, 0), H - side))
    c = img[y1:y1 + side, x1:x1 + side]
    if c.size == 0:
        return None
    c = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return clahe.apply(c)


df = assign_split(collect_teeth())
for s in ["train", "val", "test"]:
    os.makedirs(f"{OUT}/{s}", exist_ok=True)

df["path"], skipped = None, set()
for image, grp in df.groupby("image"):
    img = cv2.imread(grp.img_path.iloc[0], cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (grp.height.iloc[0], grp.width.iloc[0]):
        skipped.add(image)
        continue
    for r in grp.itertuples():
        crop = crop_tooth(img, r)
        if crop is None:
            continue
        p = f"{OUT}/{r.split}/{image[:-4]}_{r.fdi}.png"
        cv2.imwrite(p, crop)
        df.at[r.Index, "path"] = p

df = df[df.path.notna()].drop(columns=["img_path"])
df.to_csv(f"{OUT}/manifest.csv", index=False)

print("radiografias por split:\n", df.groupby("split")["image"].nunique())
print("\ndentes por split:\n", df.groupby("split").size())
print("\nrecortes positivos por classe:\n", df.groupby("split")[CLASSES].sum())
print("\ndentes com 2+ rótulos:", int((df[CLASSES].sum(axis=1) > 1).sum()))
print("imagens puladas (tamanho divergente):", sorted(skipped))