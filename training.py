import pandas as pd
import os

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


# =========================
# FOLDER OUTPUT
# =========================
OUTPUT_DIR = "output/training_result_databaru"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# PARAMETER UTAMA
# =========================
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.4
MIN_LIFT = 1.0


# =========================
# SKENARIO TRAINING
# =========================
skenario_training = {
    "60_40": "dataset/Dataset Baru/data_training6040.xlsx",
    "70_30": "dataset/Dataset Baru/data_training7030.xlsx",
    "80_20": "dataset/Dataset Baru/data_training8020.xlsx"
}


# =========================
# FUNCTION
# =========================
def format_frozenset_to_string(value):
    if isinstance(value, (set, frozenset)):
        return ", ".join(sorted(list(value)))
    return str(value)


def categorize_rule(row):
    if row["confidence"] >= 0.55 and row["lift"] >= 1.3:
        return "Strong Pattern"
    elif row["confidence"] >= 0.4 and row["lift"] > 1.0:
        return "Moderate Pattern"
    else:
        return "Weak Pattern"


def classify_rule_type(antecedents, consequents):
    all_items = list(antecedents) + list(consequents)

    has_produk = any(str(item).startswith("produk_") for item in all_items)
    has_operator = any(str(item).startswith("operator_") for item in all_items)
    has_waktu = any(str(item).startswith("waktu_") for item in all_items)

    if has_produk and not has_operator and not has_waktu:
        return "produk_produk"

    if has_produk and has_operator and has_waktu:
        return "produk_operator_waktu"

    if has_produk and has_operator:
        return "produk_operator"

    if has_produk and has_waktu:
        return "produk_waktu"

    if has_operator and has_waktu:
        return "operator_waktu"

    return "lainnya"


def label_jenis_rule(jenis_rule):
    labels = {
        "produk_produk": "Produk × Produk",
        "produk_operator": "Produk × Operator",
        "operator_waktu": "Operator × Waktu",
        "produk_waktu": "Produk × Waktu",
        "produk_operator_waktu": "Produk × Operator × Waktu",
        "lainnya": "Lainnya",
    }

    return labels.get(jenis_rule, "Lainnya")


def buat_basket(df):
    df = df.copy()

    required_columns = [
        "no_transaksi",
        "detail_produk",
        "operator",
        "kategori_waktu",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan pada data training: "
            + ", ".join(missing_columns)
        )

    df["item_produk"] = "produk_" + df["detail_produk"].astype(str).str.strip()
    df["item_operator"] = "operator_" + df["operator"].astype(str).str.strip()
    df["item_waktu"] = "waktu_" + df["kategori_waktu"].astype(str).str.strip()

    basket = df.groupby("no_transaksi").apply(
        lambda x: list(dict.fromkeys(
            x["item_produk"].dropna().tolist() +
            x["item_operator"].dropna().tolist() +
            x["item_waktu"].dropna().tolist()
        ))
    ).reset_index(name="itemset")

    return basket


# =========================
# MAIN PROGRAM
# =========================
for nama_skenario, input_file in skenario_training.items():
    print("\n==============================")
    print(f"TRAINING SKENARIO {nama_skenario}")
    print("==============================")

    if not os.path.exists(input_file):
        print(f"File tidak ditemukan: {input_file}")
        continue

    df_train = pd.read_excel(input_file)

    print("File input:", input_file)
    print("Jumlah baris training:", len(df_train))
    print("Jumlah transaksi training:", df_train["no_transaksi"].nunique())

    basket = buat_basket(df_train)

    print("Jumlah basket transaksi:", len(basket))

    te = TransactionEncoder()
    te_data = te.fit(basket["itemset"]).transform(basket["itemset"])

    df_fp = pd.DataFrame(te_data, columns=te.columns_)

    frequent_itemsets = fpgrowth(
        df_fp,
        min_support=MIN_SUPPORT,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        print("Tidak ada frequent itemset yang ditemukan.")
        continue

    frequent_itemsets["itemsets_str"] = frequent_itemsets["itemsets"].apply(
        format_frozenset_to_string
    )

    frequent_itemsets["jumlah_item"] = frequent_itemsets["itemsets"].apply(len)

    frequent_itemsets = frequent_itemsets.sort_values(
        by="support",
        ascending=False
    ).reset_index(drop=True)

    print("Jumlah frequent itemset:", len(frequent_itemsets))

    rules = association_rules(
        frequent_itemsets.drop(columns=["itemsets_str", "jumlah_item"]),
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    if rules.empty:
        print("Tidak ada association rules yang ditemukan.")
        continue

    rules["antecedents_str"] = rules["antecedents"].apply(
        format_frozenset_to_string
    )

    rules["consequents_str"] = rules["consequents"].apply(
        format_frozenset_to_string
    )

    rules["jumlah_item"] = (
        rules["antecedents"].apply(len) +
        rules["consequents"].apply(len)
    )

    rules["kategori_rule"] = rules.apply(categorize_rule, axis=1)

    # Tambahan untuk mengenali jenis rule, termasuk Produk × Produk
    rules["jenis_rule"] = rules.apply(
        lambda row: classify_rule_type(row["antecedents"], row["consequents"]),
        axis=1
    )

    rules["jenis_rule_label"] = rules["jenis_rule"].apply(label_jenis_rule)

    rules = rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    output_rules = f"{OUTPUT_DIR}/rules_training_{nama_skenario}.xlsx"
    rules.to_excel(output_rules, index=False)

    rules_filtered = rules[
        (rules["lift"] > MIN_LIFT) &
        (rules["confidence"] >= MIN_CONFIDENCE)
    ].copy()

    rules_filtered = rules_filtered.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    output_rules_filtered = f"{OUTPUT_DIR}/rules_training_filtered_{nama_skenario}.xlsx"
    rules_filtered.to_excel(output_rules_filtered, index=False)

    jumlah_produk_produk = len(
        rules_filtered[rules_filtered["jenis_rule"] == "produk_produk"]
    )

    jumlah_produk_operator = len(
        rules_filtered[rules_filtered["jenis_rule"] == "produk_operator"]
    )

    jumlah_operator_waktu = len(
        rules_filtered[rules_filtered["jenis_rule"] == "operator_waktu"]
    )

    jumlah_produk_waktu = len(
        rules_filtered[rules_filtered["jenis_rule"] == "produk_waktu"]
    )

    jumlah_produk_operator_waktu = len(
        rules_filtered[rules_filtered["jenis_rule"] == "produk_operator_waktu"]
    )

    jumlah_lainnya = len(
        rules_filtered[rules_filtered["jenis_rule"] == "lainnya"]
    )

    print("Jumlah semua rules:", len(rules))
    print("Jumlah rules setelah filter:", len(rules_filtered))

    print("\nJumlah rules per jenis:")
    print("Produk × Produk:", jumlah_produk_produk)
    print("Produk × Operator:", jumlah_produk_operator)
    print("Operator × Waktu:", jumlah_operator_waktu)
    print("Produk × Waktu:", jumlah_produk_waktu)
    print("Produk × Operator × Waktu:", jumlah_produk_operator_waktu)
    print("Lainnya:", jumlah_lainnya)

    print("\nFile semua rules disimpan:", output_rules)
    print("File rules filtered disimpan:", output_rules_filtered)

    print("\nTop 10 rules filtered:")
    print(
        rules_filtered[
            [
                "antecedents_str",
                "consequents_str",
                "support",
                "confidence",
                "lift",
                "kategori_rule",
                "jenis_rule_label"
            ]
        ].head(10)
    )

print("\n=== TRAINING SELESAI ===")