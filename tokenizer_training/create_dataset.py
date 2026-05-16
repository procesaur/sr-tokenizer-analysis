from datasets import load_dataset
from tqdm import tqdm
from json import dump, load, dumps, loads
from sklearn.model_selection import train_test_split



def clean(text):
    text = text.replace("<s>", "").replace("</s>", "")
    # text = convert(text)
    return text

def build_set(name, txt=False):
    counts = 0
    records = []
    if txt:
        fname = f"{name}.txt"
    else:
        fname = f"{name}.jsonl"

    with open(fname, "r", encoding="utf-8") as f:

        if txt:
            ds = (line for line in f)
        else:
            ds = (loads(line).get("text", "") for line in f)
        
        for text in tqdm(ds):
            if text:
                counts+=1
                records.append({"id": f"{name}_{counts}", "text": clean(text)})

        if not records:
            print(f"No records for {name}, skipping.")
            return

        # Split into train/test
        train_records, test_records = train_test_split(
            records, test_size=0.25, random_state=23
        )

        # Define file paths
        train_path =  f"{name}_train.jsonl"
        test_path = f"{name}_test.jsonl"

        # Write train
        with open(train_path, "w", encoding="utf-8") as f:
            for rec in train_records:
                f.write(dumps(rec, ensure_ascii=False) + "\n")

        # Write test
        with open(test_path, "w", encoding="utf-8") as f:
            for rec in test_records:
                f.write(dumps(rec, ensure_ascii=False) + "\n")

        print(f"{name}: wrote {len(train_records)} train and {len(test_records)} test records.")


build_set("wiki")
build_set("pdrs", True)
build_set("enauka")
build_set("nardus")
