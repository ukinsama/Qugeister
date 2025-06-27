
import os
import json
import glob
import pandas as pd
from collections import defaultdict

# === 設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_CSV = os.path.join(BASE_DIR, "elo_rating.csv")

INITIAL_ELO = 1500
K = 32

# Elo更新関数
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k=K):
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * ((1 - score_a) - expected_b)
    return new_rating_a, new_rating_b

# モデルの初期レーティング
elo_ratings = defaultdict(lambda: INITIAL_ELO)
model_ids = set()

# ログ処理
log_paths = glob.glob(os.path.join(LOG_DIR, "*/*.json"))
for path in log_paths:
    with open(path, 'r') as f:
        data = json.load(f)
        a = data["model_A"]["id"]
        b = data["model_B"]["id"]
        winner = data["result"]["winner"]
        model_ids.update([a, b])

        if winner == "model_A":
            elo_ratings[a], elo_ratings[b] = update_elo(elo_ratings[a], elo_ratings[b], 1)
        elif winner == "model_B":
            elo_ratings[b], elo_ratings[a] = update_elo(elo_ratings[b], elo_ratings[a], 1)
        elif winner == "Draw":
            elo_ratings[a], elo_ratings[b] = update_elo(elo_ratings[a], elo_ratings[b], 0.5)

# レーティング表出力
df = pd.DataFrame([
    {"Model": model, "EloRating": round(score, 2)}
    for model, score in elo_ratings.items()
]).sort_values(by="EloRating", ascending=False)

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ EloレーティングをCSV出力 → {OUTPUT_CSV}")
