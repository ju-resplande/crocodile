from datasets import Dataset

from tqdm import tqdm
import json_lines

data = {"id": [], "title": [], "context": [], "triplet": []}
with open("pt.jsonl", "rb") as f:
    for item_idx, item in enumerate(tqdm(json_lines.reader(f), total=623037)):
        for triplet_idx, triplet in enumerate(item["triples"]):
            data["id"].append(item["uri"] + "-" + str(triplet_idx + 1))
            data["title"].append(item["title"])
            data["context"].append(item["text"])
            data["triplet"].append(
                "<triplet> "
                + triplet["subject"]["surfaceform"]
                + " <subj> "
                + triplet["object"]["surfaceform"]
                + " <obj> "
                + triplet["predicate"]["surfaceform"]
            )

dataset = Dataset.from_dict(data).train_test_split(test_size=0.15)
dataset = {"validation": dataset["test"], "train": dataset["train"]}
data = dataset["train"].train_test_split(test_size=0.18)
dataset["test"] = data["test"]
dataset["train"] = data["train"]
print(dataset)
