import urllib.request
import json
import pandas as pd
from pathlib import Path

ROOT = Path("c:/Project Data/Quantara_AI_Audited_Data")
raw_save_path = ROOT / "data" / "raw" / "amfi_nippon_liquid_fund_118701_raw.json"
csv_save_path = ROOT / "data" / "raw" / "nippon_india_liquid_fund_nav_history.csv"

url = "https://api.mfapi.in/mf/118701"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=15) as resp:
    raw_text = resp.read().decode("utf-8")

# Preserve raw source unchanged
with open(raw_save_path, "w", encoding="utf-8") as f:
    f.write(raw_text)
print(f"Preserved raw JSON to {raw_save_path} ({len(raw_text)} bytes)")

parsed = json.loads(raw_text)
meta = parsed["meta"]
data = parsed["data"]

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
df["nav"] = df["nav"].astype(float)
df = df.sort_values("date").reset_index(drop=True)

df.to_csv(csv_save_path, index=False)
d_min = df["date"].min().strftime("%Y-%m-%d")
d_max = df["date"].max().strftime("%Y-%m-%d")
print(f"Saved parsed NAV CSV to {csv_save_path} ({len(df)} rows)")
print(f"Date range: {d_min} to {d_max}")
print(f"NAV range: {df['nav'].min()} to {df['nav'].max()}")

