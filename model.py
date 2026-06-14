import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json

from mlxtend.frequent_patterns import fpgrowth, association_rules


# file input & output
INPUT_FILE = "output/dataset_fp_growth.xlsx"

OUTPUT_FREQUENT_ITEMSETS = "output/frequent_itemsets.xlsx"
OUTPUT_ASSOCIATION_RULES = "output/hasil_association_rules.xlsx"
OUTPUT_ASSOCIATION_RULES_FILTERED = "output/hasil_association_rules_filtered.xlsx"
OUTPUT_ANOMALY_RULES = "output/hasil_anomali.xlsx"

# output gambar
OUTPUT_TOP_ITEM_CHART = "output/images/grafik_top_produk.png"
OUTPUT_TOP_RULE_CHART = "output/images/grafik_top_rules.png"
OUTPUT_SCATTER_RULE_CHART = "output/images/grafik_confidence_lift.png"
OUTPUT_NETWORK_GRAPH = "output/images/network_association_rules.png"

# folder output
os.makedirs("output", exist_ok=True)
os.makedirs("output/images", exist_ok=True)
os.makedirs("output/json", exist_ok=True)

# PARAMETER UTAMA
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.4
MIN_LIFT = 1.0

# FUNCTION
def convert_to_bool(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col].astype(bool)
    return df


def format_frozenset_to_string(value) -> str:
    if isinstance(value, (set, frozenset)):
        return ", ".join(sorted(list(value)))
    return str(value)


def categorize_rule(row: pd.Series) -> str:
    if row["confidence"] >= 0.55 and row["lift"] >= 1.3:
        return "Strong Pattern"
    elif row["confidence"] >= 0.4 and row["lift"] > 1.0:
        return "Moderate Pattern"
    else:
        return "Weak Pattern"


def save_rules_json(dataframe, filename):
    hasil = []

    for _, row in dataframe.iterrows():
        hasil.append({
            "rule": row["antecedents_str"] + " → " + row["consequents_str"],
            "antecedent": row["antecedents_str"],
            "consequent": row["consequents_str"],
            "support": round(row["support"], 4),
            "confidence": round(row["confidence"], 4),
            "lift": round(row["lift"], 4),
            "kategori_rule": row["kategori_rule"],
            "is_anomaly": bool(row["is_anomaly"])
        })

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=4, ensure_ascii=False)


# LOAD DATA
print("---Load Data---")
df_fp = pd.read_excel(INPUT_FILE)
print("Ukuran dataset:", df_fp.shape)

df_fp = convert_to_bool(df_fp)

print("\n---Informasi Dataset---")
print(df_fp.info())


# FP-GROWTH
print("\n---Proses FP-Growth---")

frequent_itemsets = fpgrowth(
    df_fp,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

if frequent_itemsets.empty:
    print("Tidak ada frequent itemset yang ditemukan.")
    exit()

frequent_itemsets["itemsets_str"] = frequent_itemsets["itemsets"].apply(
    format_frozenset_to_string
)

frequent_itemsets["jumlah_item"] = frequent_itemsets["itemsets"].apply(len)

frequent_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
).reset_index(drop=True)

print("Jumlah frequent itemset:", len(frequent_itemsets))
print(frequent_itemsets.head(10))

frequent_itemsets.to_excel(OUTPUT_FREQUENT_ITEMSETS, index=False)


# ASSOCIATION RULES
print("\n---Proses Association Rules---")

rules = association_rules(
    frequent_itemsets.drop(columns=["itemsets_str", "jumlah_item"]),
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

if rules.empty:
    print("Tidak ada association rules yang ditemukan.")
    exit()

rules["antecedents_str"] = rules["antecedents"].apply(format_frozenset_to_string)
rules["consequents_str"] = rules["consequents"].apply(format_frozenset_to_string)

rules["jumlah_item"] = (
    rules["antecedents"].apply(len) +
    rules["consequents"].apply(len)
)

rules_3_variabel = rules[
    (rules["jumlah_item"] >= 3) &
    (rules["antecedents_str"].str.contains("produk_", case=False, na=False)) &
    (rules["antecedents_str"].str.contains("operator_", case=False, na=False)) &
    (rules["consequents_str"].str.contains("waktu_", case=False, na=False))
].copy()

print("\n---Association Rules dengan 3 Variabel---")
print(rules_3_variabel.head())

rules_3_variabel.to_excel("output/rules_3_variabel.xlsx", index=False)


# DETEKSI ANOMALI DENGAN IQR
print("\n---Deteksi Anomali dengan IQR---")

q1_lift = rules["lift"].quantile(0.25)
q3_lift = rules["lift"].quantile(0.75)
iqr_lift = q3_lift - q1_lift

q1_conf = rules["confidence"].quantile(0.25)
q3_conf = rules["confidence"].quantile(0.75)
iqr_conf = q3_conf - q1_conf

lower_lift = q1_lift - 1.5 * iqr_lift
upper_lift = q3_lift + 1.5 * iqr_lift

lower_conf = q1_conf - 1.5 * iqr_conf
upper_conf = q3_conf + 1.5 * iqr_conf

rules["is_anomaly"] = (
    (rules["lift"] < lower_lift) |
    (rules["lift"] > upper_lift) |
    (rules["confidence"] < lower_conf) |
    (rules["confidence"] > upper_conf)
)

rules["kategori_rule"] = rules.apply(categorize_rule, axis=1)

rules = rules.sort_values(
    by=["lift", "confidence", "support"],
    ascending=[False, False, False]
).reset_index(drop=True)

print("Jumlah association rules:", len(rules))
print("Jumlah anomali:", int(rules["is_anomaly"].sum()))

print(
    rules[
        [
            "antecedents_str",
            "consequents_str",
            "support",
            "confidence",
            "lift",
            "kategori_rule",
            "is_anomaly"
        ]
    ].head(10)
)

rules.to_excel(OUTPUT_ASSOCIATION_RULES, index=False)


# FILTER RULE
print("\n---Filter Rule Association---")

rules_filtered = rules[
    (rules["lift"] > MIN_LIFT) &
    (rules["confidence"] >= MIN_CONFIDENCE)
].copy()

rules_filtered = rules_filtered.sort_values(
    by=["lift", "confidence", "support"],
    ascending=[False, False, False]
).reset_index(drop=True)

print("Jumlah rules setelah filter:", len(rules_filtered))

rules_filtered.to_excel(OUTPUT_ASSOCIATION_RULES_FILTERED, index=False)

print("Jumlah rules awal :", len(rules))
print("Jumlah rules filter :", len(rules_filtered))


# ANALISIS RULE PER VARIABEL
print("\n---Analisis Rule per Variabel---")

rules_produk_produk = rules_filtered[
    (rules_filtered["antecedents_str"].str.contains("produk_", case=False, na=False)) &
    (rules_filtered["consequents_str"].str.contains("produk_", case=False, na=False)) &
    (~rules_filtered["antecedents_str"].str.contains("operator_|waktu_", case=False, na=False)) &
    (~rules_filtered["consequents_str"].str.contains("operator_|waktu_", case=False, na=False))
].copy()

rules_produk_waktu = rules_filtered[
    (rules_filtered["antecedents_str"].str.contains("produk_", case=False, na=False)) &
    (rules_filtered["consequents_str"].str.contains("waktu_", case=False, na=False))
].copy()

rules_produk_operator = rules_filtered[
    (rules_filtered["antecedents_str"].str.contains("produk_", case=False, na=False)) &
    (rules_filtered["consequents_str"].str.contains("operator_", case=False, na=False))
].copy()

rules_operator_waktu = rules_filtered[
    (rules_filtered["antecedents_str"].str.contains("operator_", case=False, na=False)) &
    (rules_filtered["consequents_str"].str.contains("waktu_", case=False, na=False))
].copy()

rules_waktu_operator = rules_filtered[
    (rules_filtered["antecedents_str"].str.contains("waktu_", case=False, na=False)) &
    (rules_filtered["consequents_str"].str.contains("operator_", case=False, na=False))
].copy()

rules_produk_operator_waktu = rules_filtered[
    (
        rules_filtered["antecedents_str"].str.contains("produk_", case=False, na=False) |
        rules_filtered["consequents_str"].str.contains("produk_", case=False, na=False)
    ) &
    (
        rules_filtered["antecedents_str"].str.contains("operator_", case=False, na=False) |
        rules_filtered["consequents_str"].str.contains("operator_", case=False, na=False)
    ) &
    (
        rules_filtered["antecedents_str"].str.contains("waktu_", case=False, na=False) |
        rules_filtered["consequents_str"].str.contains("waktu_", case=False, na=False)
    )
].copy()

save_rules_json(rules_produk_produk, "output/json/rules_produk_produk.json")
save_rules_json(rules_produk_waktu, "output/json/rules_produk_waktu.json")
save_rules_json(rules_produk_operator, "output/json/rules_produk_operator.json")
save_rules_json(rules_operator_waktu, "output/json/rules_operator_waktu.json")
save_rules_json(rules_waktu_operator, "output/json/rules_waktu_operator.json")
save_rules_json(rules_produk_operator_waktu, "output/json/rules_produk_operator_waktu.json")

print("Jumlah rules produk → produk:", len(rules_produk_produk))
print("Jumlah rules produk → waktu:", len(rules_produk_waktu))
print("Jumlah rules produk → operator:", len(rules_produk_operator))
print("Jumlah rules operator → waktu:", len(rules_operator_waktu))
print("Jumlah rules waktu → operator:", len(rules_waktu_operator))
print("Jumlah rules produk/operator/waktu:", len(rules_produk_operator_waktu))


# RULE ANOMALI
print("---Rule Anomali---")

rules_anomaly = rules[rules["is_anomaly"] == True].copy()

rules_anomaly = rules_anomaly.sort_values(
    by=["lift", "confidence", "support"],
    ascending=[True, True, False]
).reset_index(drop=True)

print("Jumlah association rules:", len(rules))
print("Jumlah rules anomali:", len(rules_anomaly))

rules_anomaly.to_excel(OUTPUT_ANOMALY_RULES, index=False)


# VISUALISASI 1: TOP PRODUK
print("\n---Visualisasi Top Produk---")

single_items = frequent_itemsets[frequent_itemsets["jumlah_item"] == 1].copy()
single_items = single_items[
    ~single_items["itemsets_str"].str.contains("metode|waktu|operator", case=False, na=False)
]

top_items = single_items.sort_values(
    by="support",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_items,
    x="support",
    y="itemsets_str"
)

plt.title("Top 10 Produk Berdasarkan Support")
plt.xlabel("Support")
plt.ylabel("Produk")
plt.tight_layout()
plt.savefig(OUTPUT_TOP_ITEM_CHART, dpi=300)
plt.show()


# VISUALISASI 2: TOP ASSOCIATION RULES
print("\n---Visualisasi Top Rules---")

top_rules = rules_filtered.sort_values(
    by="lift",
    ascending=False
).head(10).copy()

top_rules["rule_label"] = (
    top_rules["antecedents_str"] + " → " + top_rules["consequents_str"]
)

plt.figure(figsize=(12, 7))
sns.barplot(
    data=top_rules,
    x="lift",
    y="rule_label"
)

plt.title("Top 10 Association Rules Berdasarkan Lift")
plt.xlabel("Lift")
plt.ylabel("Rules")
plt.tight_layout()
plt.savefig(OUTPUT_TOP_RULE_CHART, dpi=300)
plt.show()


# VISUALISASI 3: SCATTER CONFIDENCE VS LIFT
print("\n---Visualisasi Confidence vs Lift---")

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=rules,
    x="confidence",
    y="lift",
    hue="is_anomaly"
)

plt.title("Distribusi Association Rules Berdasarkan Confidence dan Lift")
plt.xlabel("Confidence")
plt.ylabel("Lift")
plt.tight_layout()
plt.savefig(OUTPUT_SCATTER_RULE_CHART, dpi=300)
plt.show()


# VISUALISASI 4: NETWORK GRAPH
print("\n---Visualisasi Network Graph---")

top_network_rules = rules_filtered.sort_values(
    by="lift",
    ascending=False
).head(15)

G = nx.DiGraph()

for _, row in top_network_rules.iterrows():
    G.add_edge(
        row["antecedents_str"],
        row["consequents_str"],
        weight=row["lift"]
    )

plt.figure(figsize=(14, 9))
pos = nx.spring_layout(G, k=1)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2500,
    font_size=8,
    arrows=True
)

plt.title("Network Graph Association Rules")
plt.tight_layout()
plt.savefig(OUTPUT_NETWORK_GRAPH, dpi=300)
plt.show()


print("\n---Save JSON Utama---")

frequent_itemsets.to_json(
    "output/json/frequent_itemsets.json",
    orient="records",
    indent=4
)

rules.to_json(
    "output/json/hasil_association_rules.json",
    orient="records",
    indent=4
)

rules_filtered.to_json(
    "output/json/hasil_association_rules_filtered.json",
    orient="records",
    indent=4
)

rules_anomaly.to_json(
    "output/json/hasil_anomali.json",
    orient="records",
    indent=4
)


# 1. BAR CHART TOP 10 PRODUK
single_produk = frequent_itemsets[
    (frequent_itemsets["jumlah_item"] == 1) &
    (~frequent_itemsets["itemsets_str"].str.contains("metode_|waktu_|operator_", case=False, na=False))
].copy()

top_produk = single_produk.sort_values(
    by="support",
    ascending=False
).head(10)

top_produk_json = []

for _, row in top_produk.iterrows():
    top_produk_json.append({
        "produk": row["itemsets_str"].replace("produk_", ""),
        "support": round(row["support"], 4)
    })

with open("output/json/top_10_produk.json", "w", encoding="utf-8") as f:
    json.dump(top_produk_json, f, indent=4, ensure_ascii=False)


# 2. BAR CHART TOP 10 RULES
top_rules = rules_filtered.sort_values(
    by="lift",
    ascending=False
).head(10)

top_rules_json = []

for _, row in top_rules.iterrows():
    top_rules_json.append({
        "rule": row["antecedents_str"] + " → " + row["consequents_str"],
        "support": round(row["support"], 4),
        "confidence": round(row["confidence"], 4),
        "lift": round(row["lift"], 4),
        "kategori_rule": row["kategori_rule"],
        "is_anomaly": bool(row["is_anomaly"])
    })

with open("output/json/top_10_rules.json", "w", encoding="utf-8") as f:
    json.dump(top_rules_json, f, indent=4, ensure_ascii=False)


# 3. HEATMAP CONFIDENCE DAN LIFT
heatmap_conf_lift = rules_filtered.copy()

heatmap_conf_lift_json = []

for _, row in heatmap_conf_lift.iterrows():
    heatmap_conf_lift_json.append({
        "antecedent": row["antecedents_str"],
        "consequent": row["consequents_str"],
        "confidence": round(row["confidence"], 4),
        "lift": round(row["lift"], 4)
    })

with open("output/json/heatmap_confidence_lift.json", "w", encoding="utf-8") as f:
    json.dump(heatmap_conf_lift_json, f, indent=4, ensure_ascii=False)


# 4. DISTRIBUSI ASSOCIATION RULES
distribusi_rules = rules["kategori_rule"].value_counts().reset_index()
distribusi_rules.columns = ["kategori_rule", "jumlah"]

distribusi_rules_json = []

for _, row in distribusi_rules.iterrows():
    distribusi_rules_json.append({
        "kategori_rule": row["kategori_rule"],
        "jumlah": int(row["jumlah"])
    })

with open("output/json/distribusi_association_rules.json", "w", encoding="utf-8") as f:
    json.dump(distribusi_rules_json, f, indent=4, ensure_ascii=False)


print("File JSON selesai")
print("\nFinish")