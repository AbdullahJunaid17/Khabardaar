import os, json, glob
import pandas as pd

folder_path = "C:/Users/HP/Khabardaar/data/jsons"   # your exact path

records = []
for file in glob.glob(os.path.join(folder_path, "*.json")):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data.get("content_original", "")
    label = data.get("bias", -1)
    
    # Only keep if both text and valid label exist
    if text and label in [0, 1, 2]:
        records.append({"text": text, "label": int(label)})
        print("1")

df = pd.DataFrame(records)
print("Loaded:", len(df), "articles")
print(df['label'].value_counts())
print(df.head())

df.to_csv("C:/Users/HP/Khabardaar/data/combined_bias.csv", index=False)
print("Saved combined_bias.csv")
