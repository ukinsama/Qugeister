
import os
import json
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns

# === 設定 ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # quantum_match_eval/
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_CSV = os.path.join(BASE_DIR, "nash_rating.csv")
OUTPUT_PNG = os.path.join(BASE_DIR, "nash_rating_top10.png")

# === 1. ログ読み込みと勝率行列構築 ===
win_count = defaultdict(lambda: defaultdict(int))
match_count = defaultdict(lambda: defaultdict(int))

for path in glob.glob(os.path.join(LOG_DIR, "*/*.json")):
    with open(path, "r") as f:
        data = json.load(f)
        a = data["model_A"]["id"]
        b = data["model_B"]["id"]
        winner = data["result"]["winner"]
        match_count[a][b] += 1
        if winner == "model_A":
            win_count[a][b] += 1
        elif winner == "model_B":
            win_count[b][a] += 1

model_ids = sorted(set(list(win_count.keys()) + list(match_count.keys())))
winrate_matrix = []
for i in model_ids:
    row = []
    for j in model_ids:
        total = match_count[i][j] + match_count[j][i]
        row.append(win_count[i][j] / total if total > 0 else 0.5)
    winrate_matrix.append(row)

W = np.array(winrate_matrix)
n = len(model_ids)
W = W.reshape((n, n))  # 明示的に2次元正方行列に

# === 2. ナッシュ均衡計算（最大最小戦略）===
c = np.zeros(n + 1)
c[-1] = -1
A_ub = np.hstack([-W.T, np.ones((n, 1))])
b_ub = np.zeros(n)
A_eq = np.zeros((1, n + 1))
A_eq[0, :n] = 1
b_eq = [1]
bounds = [(0, 1)] * n + [(0, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if not res.success:
    raise RuntimeError("ナッシュ均衡の計算に失敗しました。")

strategy = res.x[:n]
ratings = W @ strategy

# === 3. データフレーム作成・保存 ===
rating_table = pd.DataFrame({
    "Model": model_ids,
    "NashRating": np.round(ratings, 4),
    "MixStrategyProb": np.round(strategy, 4)
}).sort_values(by="NashRating", ascending=False).reset_index(drop=True)

rating_table.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Nash Rating CSV 出力: {OUTPUT_CSV}")

# === 4. 可視化 ===
top_n = 10
top_models = rating_table.head(top_n)

plt.figure(figsize=(10, 6))
sns.barplot(x="NashRating", y="Model", data=top_models, palette="viridis")
plt.title(f"Top {top_n} Models by Nash Equilibrium Rating")
plt.xlabel("Nash Rating (Expected Win Rate)")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
print(f"📊 可視化PNG出力: {OUTPUT_PNG}")
plt.show()
