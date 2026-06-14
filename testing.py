import pandas as pd
import os

os.makedirs("output/testing_result_databaru", exist_ok=True)

MIN_CONFIDENCE = 0.4
MIN_LIFT = 1.0

skenario_testing = {
    "60_40": {
        "rules": "output/training_result_databaru/rules_training_filtered_60_40.xlsx",
        "testing": "dataset/Split data testing/data_testing6040.xlsx"
    },
    "70_30": {
        "rules": "output/training_result_databaru/rules_training_filtered_70_30.xlsx",
        "testing": "dataset/Split data testing/data_testing7030.xlsx"
    },
    "80_20": {
        "rules": "output/training_result_databaru/rules_training_filtered_80_20.xlsx",
        "testing": "dataset/Split data testing/data_testing8020.xlsx"
    }
}


def split_items(text):
    if pd.isna(text) or text == "":
        return set()

    return set([
        item.strip()
        for item in str(text).split(",")
        if item.strip() != ""
    ])


def categorize_rule_testing(confidence, lift):
    if confidence >= 0.55 and lift >= 1.3:
        return "Strong Pattern"
    elif confidence >= 0.4 and lift > 1.0:
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


def buat_basket_testing(df):
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
            "Kolom wajib tidak ditemukan pada data testing: "
            + ", ".join(missing_columns)
        )

    df["item_produk"] = "produk_" + df["detail_produk"].astype(str).str.strip()
    df["item_operator"] = "operator_" + df["operator"].astype(str).str.strip()
    df["item_waktu"] = "waktu_" + df["kategori_waktu"].astype(str).str.strip()

    basket = df.groupby("no_transaksi").apply(
        lambda x: set(
            x["item_produk"].dropna().tolist() +
            x["item_operator"].dropna().tolist() +
            x["item_waktu"].dropna().tolist()
        )
    ).reset_index(name="itemset")

    return basket


ringkasan_testing = []

for nama_skenario, file_path in skenario_testing.items():
    print("\n-------------------------------")
    print(f"Testing Skenario {nama_skenario}")
    print("-------------------------------")

    if not os.path.exists(file_path["rules"]):
        print("File rules tidak ditemukan:", file_path["rules"])
        continue

    if not os.path.exists(file_path["testing"]):
        print("File testing tidak ditemukan:", file_path["testing"])
        continue

    rules_training = pd.read_excel(file_path["rules"])
    df_testing = pd.read_excel(file_path["testing"])

    print("File rules:", file_path["rules"])
    print("File testing:", file_path["testing"])
    print("Jumlah rules training yang diuji:", len(rules_training))
    print("Jumlah baris data testing:", len(df_testing))
    print("Jumlah transaksi testing:", df_testing["no_transaksi"].nunique())

    basket_testing = buat_basket_testing(df_testing)
    jumlah_transaksi_testing = len(basket_testing)

    hasil_testing = []

    for _, rule in rules_training.iterrows():
        antecedent = split_items(rule["antecedents_str"])
        consequent = split_items(rule["consequents_str"])

        jenis_rule = classify_rule_type(antecedent, consequent)
        jenis_rule_label = label_jenis_rule(jenis_rule)

        count_antecedent = 0
        count_consequent = 0
        count_both = 0

        for itemset in basket_testing["itemset"]:
            if antecedent.issubset(itemset):
                count_antecedent += 1

            if consequent.issubset(itemset):
                count_consequent += 1

            if antecedent.issubset(itemset) and consequent.issubset(itemset):
                count_both += 1

        antecedent_support = count_antecedent / jumlah_transaksi_testing if jumlah_transaksi_testing > 0 else 0
        consequent_support = count_consequent / jumlah_transaksi_testing if jumlah_transaksi_testing > 0 else 0
        support = count_both / jumlah_transaksi_testing if jumlah_transaksi_testing > 0 else 0
        confidence = count_both / count_antecedent if count_antecedent > 0 else 0
        lift = confidence / consequent_support if consequent_support > 0 else 0

        kategori_rule = categorize_rule_testing(confidence, lift)

        if confidence >= MIN_CONFIDENCE and lift > MIN_LIFT:
            status_pengujian = "Konsisten"
        else:
            status_pengujian = "Tidak Konsisten"

        hasil_testing.append({
            "Antecedent": rule["antecedents_str"],
            "Consequent": rule["consequents_str"],

            "Jenis Rule": jenis_rule,
            "Jenis Rule Label": jenis_rule_label,

            "Support Training": rule["support"],
            "Confidence Training": rule["confidence"],
            "Lift Training": rule["lift"],
            "Kategori Training": rule["kategori_rule"],

            "Antecedent Support": antecedent_support,
            "Consequent Support": consequent_support,
            "Support": support,
            "Confidence": confidence,
            "Lift": lift,
            "Kategori Rule": kategori_rule,

            "Count Antecedent": count_antecedent,
            "Count Consequent": count_consequent,
            "Count Both": count_both,

            "Status Pengujian": status_pengujian
        })

    df_hasil_testing = pd.DataFrame(hasil_testing)

    if df_hasil_testing.empty:
        print("Tidak ada hasil testing untuk skenario:", nama_skenario)
        continue

    df_hasil_testing = df_hasil_testing.sort_values(
        by=["Lift", "Confidence", "Support"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    output_testing = f"output/testing_result_databaru/evaluation_testing_{nama_skenario}.xlsx"
    df_hasil_testing.to_excel(output_testing, index=False)

    jumlah_konsisten = len(df_hasil_testing[df_hasil_testing["Status Pengujian"] == "Konsisten"])
    jumlah_tidak_konsisten = len(df_hasil_testing[df_hasil_testing["Status Pengujian"] == "Tidak Konsisten"])

    kategori_counts = df_hasil_testing["Kategori Rule"].value_counts()

    strong_count = int(kategori_counts.get("Strong Pattern", 0))
    moderate_count = int(kategori_counts.get("Moderate Pattern", 0))
    weak_count = int(kategori_counts.get("Weak Pattern", 0))

    rata_support = df_hasil_testing["Support"].mean()
    rata_confidence = df_hasil_testing["Confidence"].mean()
    rata_lift = df_hasil_testing["Lift"].mean()

    ringkasan_testing.append({
        "Skenario": nama_skenario,
        "Jumlah Rules Diuji": len(rules_training),
        "Jumlah Rules Konsisten": jumlah_konsisten,
        "Jumlah Rules Tidak Konsisten": jumlah_tidak_konsisten,
        "Strong Pattern": strong_count,
        "Moderate Pattern": moderate_count,
        "Weak Pattern": weak_count,
        "Rata-rata Support": rata_support,
        "Rata-rata Confidence": rata_confidence,
        "Rata-rata Lift": rata_lift,
    })

    print("Jumlah rules konsisten:", jumlah_konsisten)
    print("Jumlah rules tidak konsisten:", jumlah_tidak_konsisten)
    print("File hasil testing disimpan:", output_testing)

    print("\nTop 10 hasil testing:")
    print(
        df_hasil_testing[
            [
                "Antecedent",
                "Consequent",
                "Jenis Rule Label",
                "Confidence",
                "Lift",
                "Kategori Rule",
                "Status Pengujian"
            ]
        ].head(10)
    )


df_ringkasan_testing = pd.DataFrame(ringkasan_testing)

df_ringkasan_testing.to_excel(
    "output/testing_result_databaru/ringkasan_testing.xlsx",
    index=False
)

print("\n-------------------------------")
print("Testing Finish")
print(df_ringkasan_testing)