import sys
import json
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.feature_engineering.hydrologic_routing import extract_d8_streamlines

cases = [
    'case_01_amarnath_cloudburst_2022',
    'case_02_north_india_squall_2018',
    'case_03_himachal_flash_flood_2023',
    'case_04_wayanad_deluge_2024'
]

for c in cases:
    t0 = time.time()
    res = extract_d8_streamlines(c, min_accumulation=4.0, max_paths=30)
    out_path = Path('data/processed') / c / 'd8_streamlines.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    elapsed = time.time() - t0
    print(f"Precomputed {c}: {len(res.get('streamlines', []))} streamlines in {elapsed:.2f}s -> {out_path} ({out_path.stat().st_size} bytes)")
