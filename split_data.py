import random
import os

from tqdm import tqdm
import json_lines

DATA_DIR = "out_clean"


def format_triplet(triplet):
    return (
        "<triplet> "
        + triplet["subject"]["surfaceform"]
        + " <subj> "
        + triplet["object"]["surfaceform"]
        + " <obj> "
        + triplet["predicate"]["surfaceform"]
    )


splits = {
    "train": [],
    "dev": [],
    "test": [],
}

total = 623037

data = list()
with open(os.path.join(DATA_DIR, "pt.jsonl"), "rb") as f:
    for item in tqdm(json_lines.reader(f), total=total):
        data.append(item)

data = random.shuffle(data)
data["train"] = "\n".join(data[0 : int(0.70 * total)])
data["val"] = "\n".join(data[int(0.70 * total) : int(0.85 * total)])
data["test"] = "\n".join(data[int(0.85 * total) :])

for split in splits:
    with open(os.path.join(DATA_DIR, f"pt_{split}.jsonl"), "w") as f:
        f.write(data[split])
