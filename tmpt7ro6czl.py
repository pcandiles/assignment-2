import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from api_key import load_api_key

WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

WORKSPACE, PROJECT, VERSION = "product-defect-detection", "bottle-defects-o4gnx", 28
GOOD_CLASS = "Good Cap"

MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", 50))
IMGSZ = int(os.environ.get("IMGSZ", 320))
SEED = 0

# --- Cost matrix, EUR per bottle -------------------------------------------
# Chosen by the author; the reasoning for each belongs in REPORT.md.
COST_FN = 50.0      # defective bottle passed as good
COST_FP = 2.0       # good bottle rejected
COST_TP = 0.0       # defective bottle correctly rejected
COST_TN = 0.0       # good bottle correctly passed
COST_REVIEW = 0.5   # one human check (assumed to get the final call right)
BASE_RATE = 0.05    # assumed share of defective bottles in production

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

DATA = ROOT / "data" / f"{PROJECT}-v{VERSION}"
if not (DATA / "data.yaml").exists():
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(DATA)
    )

NAMES = yaml.safe_load((DATA / "data.yaml").read_text())["names"]
GOOD_IDX = NAMES.index(GOOD_CLASS)

# Roboflow's data.yaml uses relative paths that depend on the working directory.
DATA_YAML = DATA / "ds.yaml"
DATA_YAML.write_text(yaml.safe_dump({
    "path": str(DATA),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": {i: n for i, n in enumerate(NAMES)},
}))


def read_labels(split):
    """{image stem: array of [cls, x, y, w, h]} for one split."""
    out = {}
    for img in sorted((DATA / split / "images").glob("*")):
        f = DATA / split / "labels" / f"{img.stem}.txt"
        rows = [l.split() for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
        out[img] = np.array(rows, dtype=float).reshape(-1, 5)
    return out


LABELS = {s: read_labels(s) for s in ("train", "valid", "test")}

dataset_stats = {}
for split, items in LABELS.items():
    per_class = {}
    for k, name in enumerate(NAMES):
        areas = [r[3] * r[4] for rows in items.values() for r in rows if int(r[0]) == k]
        per_class[name] = {
            "instances": len(areas),
            "median_box_area_pct": round(100 * float(np.median(areas)), 1) if areas else None,
        }
    dataset_stats[split] = {"images": len(items), "classes": per_class}
(RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=1))
print(pd.DataFrame({s: {n: v["instances"] for n, v in d["classes"].items()} for s, d in dataset_stats.items()}))
print({s: d["images"] for s, d in dataset_stats.items()})

model = YOLO(MODEL)
model.train(
    data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMGSZ, seed=SEED,
    project=str(ROOT / "runs"), name="train", exist_ok=True, plots=True,
)
shutil.copy(model.trainer.best, WEIGHTS_DIR / "best.pt")
tuned = YOLO(str(WEIGHTS_DIR / "best.pt"))

val = tuned.val(
    data=str(DATA_YAML), split="test", imgsz=IMGSZ, conf=0.001, plots=True,
    project=str(ROOT / "runs"), name="val_test", exist_ok=True,
)
per_class = {}
for pos, k in enumerate(val.box.ap_class_index):
    per_class[NAMES[int(k)]] = {
        "precision": float(val.box.p[pos]),
        "recall": float(val.box.r[pos]),
        "ap50": float(val.box.ap50[pos]),
        "ap50_95": float(val.box.ap[pos]),
        "test_instances": dataset_stats["test"]["classes"][NAMES[int(k)]]["instances"],
    }
box_metrics = {
    "map50": float(val.box.map50),
    "map50_95": float(val.box.map),
    "per_class": per_class,
    "confusion_matrix": {
        "labels": NAMES + ["background"],
        "matrix": val.confusion_matrix.matrix.tolist(),
        "convention": "matrix[predicted][true]",
    },
}
print(pd.DataFrame(per_class).T.round(3))
print("mAP50", round(box_metrics["map50"], 3), "mAP50-95", round(box_metrics["map50_95"], 3))

def bottle_scores(weights, split):
    images = list(LABELS[split])
    truth = np.array([bool((LABELS[split][p][:, 0].astype(int) != GOOD_IDX).any()) for p in images])
    scores = np.zeros(len(images))
    results = weights.predict(
        source=[str(p) for p in images], conf=0.001, imgsz=IMGSZ, verbose=False, stream=True
    )
    for i, r in enumerate(results):
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        defect = cls != GOOD_IDX
        scores[i] = float(conf[defect].max()) if defect.any() else 0.0
    return pd.DataFrame({"image": [p.name for p in images], "defective": truth, "score": scores})


scores = {s: bottle_scores(tuned, s) for s in ("valid", "test")}
for s, df in scores.items():
    df.to_csv(RESULTS_DIR / f"scores_{s}.csv", index=False)
    print(s, len(df), "bottles,", int(df.defective.sum()), "defective")

def rates(y, s, lo, hi):
    """Share of defective / good bottles that end up passed, reviewed, failed."""
    fail, review = s >= hi, (s >= lo) & (s < hi)
    pos, neg = max(int(y.sum()), 1), max(int((~y).sum()), 1)
    return {
        "tp": (fail & y).sum() / pos, "rev_pos": (review & y).sum() / pos, "fn": 1 - ((fail | review) & y).sum() / pos,
        "fp": (fail & ~y).sum() / neg, "rev_neg": (review & ~y).sum() / neg, "tn": 1 - ((fail | review) & ~y).sum() / neg,
    }


def cost_per_1000(y, s, lo, hi, base_rate=None):
    """Expected EUR per 1,000 bottles. lo == hi means no review band."""
    b = BASE_RATE if base_rate is None else base_rate
    r = rates(y, s, lo, hi)
    defective = r["tp"] * COST_TP + r["rev_pos"] * (COST_REVIEW + COST_TP) + r["fn"] * COST_FN
    good = r["fp"] * COST_FP + r["rev_neg"] * (COST_REVIEW + COST_TN) + r["tn"] * COST_TN
    return 1000 * (b * defective + (1 - b) * good)


def review_share(y, s, lo, hi, base_rate=None):
    b = BASE_RATE if base_rate is None else base_rate
    r = rates(y, s, lo, hi)
    return b * r["rev_pos"] + (1 - b) * r["rev_neg"]


yv, sv = scores["valid"].defective.to_numpy(), scores["valid"].score.to_numpy()
yt, st = scores["test"].defective.to_numpy(), scores["test"].score.to_numpy()

sweep = pd.DataFrame({
    "threshold": THRESHOLDS,
    "cost_per_1000_valid": [cost_per_1000(yv, sv, t, t) for t in THRESHOLDS],
    "cost_per_1000_test": [cost_per_1000(yt, st, t, t) for t in THRESHOLDS],
    "recall_test": [rates(yt, st, t, t)["tp"] for t in THRESHOLDS],
    "false_positive_rate_test": [rates(yt, st, t, t)["fp"] for t in THRESHOLDS],
})
sweep.to_csv(RESULTS_DIR / "sweep.csv", index=False)

# The threshold is chosen on validation, then reported on test.
best_t = float(sweep.loc[sweep.cost_per_1000_valid.idxmin(), "threshold"])

# Pass / review / fail band, also chosen on validation.
band = min(
    ((lo, hi) for lo in THRESHOLDS for hi in THRESHOLDS if lo < hi),
    key=lambda p: cost_per_1000(yv, sv, p[0], p[1]),
)
print("single threshold:", best_t, "| band (pass below, fail at/above):", band)

coco = YOLO("yolo11n.pt")
coco_hits = 0
for r in coco.predict(source=[str(p) for p in LABELS["test"]], conf=0.25, imgsz=IMGSZ, verbose=False, stream=True):
    coco_hits += int(len(r.boxes) > 0)

train_defective = np.array([(rows[:, 0].astype(int) != GOOD_IDX).any() for rows in LABELS["train"].values()])
majority_is_defective = bool(train_defective.mean() >= 0.5)


def confusion(y, pred):
    return {
        "tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()),
        "fn": int((~pred & y).sum()), "tn": int((~pred & ~y).sum()),
    }


def summarise(c):
    n = sum(c.values())
    return {
        **c,
        "accuracy": (c["tp"] + c["tn"]) / n,
        "precision": c["tp"] / max(c["tp"] + c["fp"], 1),
        "recall": c["tp"] / max(c["tp"] + c["fn"], 1),
    }


all_pass = np.zeros(len(yt), dtype=bool)
bottle_level = {
    "finetuned": summarise(confusion(yt, st >= best_t)),
    "coco_zero_shot": {**summarise(confusion(yt, all_pass)), "test_images_with_any_coco_detection": coco_hits},
    "predict_nothing": summarise(confusion(yt, all_pass)),
    "majority_class": summarise(confusion(yt, np.full(len(yt), majority_is_defective))),
    "majority_class_label": "defective" if majority_is_defective else "good",
}
print(pd.DataFrame({k: v for k, v in bottle_level.items() if isinstance(v, dict)}).T.round(3))

lo, hi = band
strategies = {
    "do_nothing": 1000 * (BASE_RATE * COST_FN + (1 - BASE_RATE) * COST_TN),
    "human_checks_everything": 1000 * (BASE_RATE * (COST_REVIEW + COST_TP) + (1 - BASE_RATE) * (COST_REVIEW + COST_TN)),
    "model_alone": cost_per_1000(yt, st, best_t, best_t),
    "model_plus_human_review": cost_per_1000(yt, st, lo, hi),
}
print({k: round(v, 2) for k, v in strategies.items()}, "EUR per 1,000 bottles")

metrics = {
    "model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "seed": SEED,
    "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
    "good_class": GOOD_CLASS,
    "box": box_metrics,
    "bottle_level_test": bottle_level,
}
(RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1))

operating_point = {
    "threshold": best_t,
    "band": {"pass_below": float(lo), "fail_at_or_above": float(hi)},
    "chosen_on": "valid",
    "cost_matrix_eur": {
        "false_negative": COST_FN, "false_positive": COST_FP,
        "true_positive": COST_TP, "true_negative": COST_TN, "human_review": COST_REVIEW,
    },
    "base_rate": BASE_RATE,
    "test": {
        "cost_per_1000_threshold": cost_per_1000(yt, st, best_t, best_t),
        "cost_per_1000_band": cost_per_1000(yt, st, lo, hi),
        "review_share_band": review_share(yt, st, lo, hi),
    },
    "strategies_cost_per_1000_test": strategies,
}
(RESULTS_DIR / "operating_point.json").write_text(json.dumps(operating_point, indent=1))

assert (WEIGHTS_DIR / "best.pt").is_file()
print("Saved weights/best.pt and", sorted(p.name for p in RESULTS_DIR.iterdir()))
