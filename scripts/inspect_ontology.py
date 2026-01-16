
import pandas as pd
import ast
from collections import Counter

def main():
    manifest_path = "data/manifests/master_manifest.csv"
    df = pd.read_csv(manifest_path)
    
    # 1. Inspect Chapman Ontology
    print("\n--- Chapman Ontology (ID -> Most Common Raw Labels) ---")
    chap = df[df['dataset_source'] == 'chapman']
    for i in range(7):
        subset = chap[chap['task_a_label'] == i]
        if len(subset) == 0:
            print(f"Class {i}: Empty")
            continue
        
        # Raw labels are likely strings
        # Count most common
        c = Counter(subset['raw_labels'].dropna())
        print(f"Class {i} (n={len(subset)}): {c.most_common(5)}")

    # 2. Inspect PTB-XL SCP Codes (stored in raw_labels)
    print("\n--- PTB-XL SCP Codes Sample ---")
    ptb = df[df['dataset_source'] == 'ptbxl']
    print(f"Total PTB: {len(ptb)}")
    print(ptb['raw_labels'].dropna().head(10).values)
    
    # See what codes exist in PTB
    # Flatten all keys
    all_codes = Counter()
    for valid_json in ptb['raw_labels'].dropna():
        try:
            # It's a string representation of a dict: "{'NORM': 100, ...}"
            # Or just a list? Or string?
            if valid_json.startswith("{"):
                 d = ast.literal_eval(valid_json)
                 all_codes.update(d.keys())
            else:
                 all_codes[valid_json] += 1
        except:
            pass
            
    print("\nTop 20 PTB-XL Codes:")
    print(all_codes.most_common(20))

if __name__ == "__main__":
    main()
