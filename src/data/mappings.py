# Label Mapping Definitions

# 7-Class Standard Taxonomy
TASK_A_LABELS = {
    0: "NSR",   # Normal Sinus Rhythm
    1: "AFIB",  # Atrial Fibrillation / Flutter
    2: "GSVT",  # Supraventricular Tachycardia (or equivalent)
    3: "SB",    # Sinus Bradycardia
    4: "ST",    # Sinus Tachycardia
    5: "PVC",   # Premature Ventricular Contractions (ECT)
    6: "OTHER"  # Conducting defects, MI, Hypertrophy, etc.
}

# PTB-XL SCP Codes Mapping
# Based on 'diagnostic_class' and 'subclass' aggregation
PTBXL_MAPPING = {
    'NORM': 0,
    'AFIB': 1,
    'AFLT': 1,  # Group flutter with fib
    'SVT': 2,
    'PSVT': 2,
    'SB': 3,
    'SR': 0,    # Sinus Rhythm often NORM
    'STACH': 4,
    'PVC': 5,
    # Everything else maps to 6 (OTHER) or dropped if Noise
}

# Chapman SNOMED Codes Mapping (Common codes in the dataset)
# Using code suffixes where possible, or full strings if provided as text
CHAPMAN_MAPPING = {
    # Normal
    "426783006": 0, # Sinus Rhythm
    
    # AFIB
    "164889003": 1, # Atrial Fibrillation
    "164890007": 1, # Atrial Flutter
    
    # GSVT
    "426177001": 3, # Sinus Bradycardia (Wait, verify code) -> 426177001 is SB
    "427084000": 4, # Sinus Tachycardia
    
    "427393009": 2, # SVT
    
    # PVC/Ectopy
    "164884008": 5, # PVC
    "427172004": 5, # PVC bigeminy
    
    # Other common ones to map to OTHER (6)
    "164909002": 6, # LBBB
    "284470004": 6, # RBBB
    "164931005": 6, # ST-T changes
}

# MIT-BIH uses Beat Annotations, handled logically in script
MITBIH_MAPPING = {
    'N': 0,
    'L': 6, 'R': 6, # Bundle branch blocks -> OTHER
    'A': 2, 'a': 2, 'J': 2, 'S': 2, # SVT related
    'V': 5, 'E': 5, # PVCs
    'f': 1, # Fusion/Flutter -> AFIB logic handled by window duration
    '/': 6, # Paced -> OTHER
}
