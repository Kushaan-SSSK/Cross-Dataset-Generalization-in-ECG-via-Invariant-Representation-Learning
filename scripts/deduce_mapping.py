
import pandas as pd
import json
import ast
from collections import Counter

def main():
    manifest_path = "data/manifests/master_manifest.csv"
    df = pd.read_csv(manifest_path)
    
    # 1. Chapman Taxonomy
    print("Deducing Chapman Taxonomy...")
    chap = df[df['dataset_source'] == 'chapman']
    
    chap_map = {}
    for i in sorted(chap['task_a_label'].dropna().unique().astype(int)):
        subset = chap[chap['task_a_label'] == i]
        # raw_labels are like "SR", "AFIB", etc ? Or "['SR']"?
        # Let's count
        c = Counter()
        for x in subset['raw_labels'].dropna():
            c[x] += 1
        
        # Most common
        top = c.most_common(1)[0][0]
        chap_map[int(i)] = top
        
    # 2. PTB-XL Codes
    print("Analyzing PTB-XL Codes...")
    ptb = df[df['dataset_source'] == 'ptbxl']
    ptb_codes = Counter()
    for x in ptb['raw_labels'].dropna():
        try:
             # Assume it's a list string "['SR', 'NORM']" or dict string
             if x.startswith("["):
                 # List
                 codes = ast.literal_eval(x)
                 ptb_codes.update(codes)
             elif x.startswith("{"):
                 # Dict
                 d = ast.literal_eval(x)
                 ptb_codes.update(d.keys())
             else:
                 ptb_codes[x] += 1
        except:
             pass
             
    res = {
        "chapman_taxonomy": chap_map,
        "ptbxl_top_50": ptb_codes.most_common(50)
    }
    
    with open("mapping_deduction.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
