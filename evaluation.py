import pandas as pd
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

# FOLDER OUPUT
os.makedirs("output/evaluation_report", exist_ok=True)

# INPUT DAN OUTPUT FILE
INPUT_EVALUATION = "output/testing_result_databaru/evaluation_testing_70_30.xlsx"
INPUT_RINGKASAN = "output/testing_result_databaru/ringkasan_testing.xlsx"

OUTPUT_TOP_RULES_EXCEL = "output/evaluation_report/databaru_top_rules_evaluation_70_30.xlsx"
OUTPUT_RINGKASAN_EXCEL = "output/evaluation_report/databaru_ringkasan_evaluation_70_30.xlsx"
OUTPUT_GAMBAR_TABLE = "output/evaluation_report/databaru_gambar_evaluasi_rules_70_30.png"

# CEK FILE
if not os.path.exists(INPUT_EVALUATION):
    raise FileNotFoundError(f"File evaluation tidak ditemukan: {INPUT_EVALUATION}")

if not os.path.exists(INPUT_RINGKASAN):
    raise FileNotFoundError(f"File ringkasan tidak ditemukan: {INPUT_RINGKASAN}")

# LOAD DATA
df_eval = pd.read_excel(INPUT_EVALUATION)
df_ringkasan = pd.read_excel(INPUT_RINGKASAN)

# Ambil ringkasan skenario terbaik, yaitu 70:30
ringkasan_70_30 = df_ringkasan[df_ringkasan["Skenario"] == "70_30"].copy()

if ringkasan_70_30.empty:
    print("Data ringkasan untuk skenario 70_30 tidak ditemukan.")
    exit()

# KOLOM LAPORAN
kolom_laporan = [
    "Antecedent",
    "Consequent",
    "Antecedent Support",
    "Consequent Support",
    "Support",
    "Confidence",
    "Lift",
    "Kategori Rule"
]

kolom_laporan = [kolom for kolom in kolom_laporan if kolom in df_eval.columns]

df_laporan = df_eval[kolom_laporan].copy()

# URUTKAN BERDASARKAN CONFIDENCE DAN LIFT
df_laporan = df_laporan.sort_values(
    by=["Confidence", "Lift"],
    ascending=[False, False]
).reset_index(drop=True)

# Ambil 10 rule teratas
top_rules = df_laporan.head(10).copy()

# RAPIKAN ANGKA
kolom_angka = [
    "Antecedent Support",
    "Consequent Support",
    "Support",
    "Confidence",
    "Lift"
]

for kolom in kolom_angka:
    if kolom in top_rules.columns:
        top_rules[kolom] = top_rules[kolom].round(4)

print("\n=== TABEL EVALUASI TOP 10 RULES ===")
print(
    tabulate(
        top_rules,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

# Save to excel
top_rules.to_excel(OUTPUT_TOP_RULES_EXCEL, index=False)
ringkasan_70_30.to_excel(OUTPUT_RINGKASAN_EXCEL, index=False)

# Gambar Tabel Evaluasi
df_gambar = top_rules.head(5).copy()

kolom_gambar = [
    "Antecedent",
    "Consequent",
    "Support",
    "Confidence",
    "Lift",
    "Kategori Rule"
]

kolom_gambar = [kolom for kolom in kolom_gambar if kolom in df_gambar.columns]
df_gambar = df_gambar[kolom_gambar].copy()

plt.figure(figsize=(15, 4))
plt.axis("off")

table = plt.table(
    cellText=df_gambar.values,
    colLabels=df_gambar.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title(
    "Hasil Evaluasi Aturan Asosiasi Skenario 70:30 Berdasarkan Confidence Tertinggi",
    fontsize=12,
    pad=20
)

plt.tight_layout()
plt.savefig(OUTPUT_GAMBAR_TABLE, dpi=300, bbox_inches="tight")
plt.show()

# Ringkasan Evaluasi
jumlah_rules = int(ringkasan_70_30["Jumlah Rules Diuji"].iloc[0])
jumlah_konsisten = int(ringkasan_70_30["Jumlah Rules Konsisten"].iloc[0])
jumlah_tidak_konsisten = int(ringkasan_70_30["Jumlah Rules Tidak Konsisten"].iloc[0])

strong = int(ringkasan_70_30["Strong Pattern"].iloc[0])
moderate = int(ringkasan_70_30["Moderate Pattern"].iloc[0])
weak = int(ringkasan_70_30["Weak Pattern"].iloc[0])

rata_support = float(ringkasan_70_30["Rata-rata Support"].iloc[0])
rata_confidence = float(ringkasan_70_30["Rata-rata Confidence"].iloc[0])
rata_lift = float(ringkasan_70_30["Rata-rata Lift"].iloc[0])

print("\n----Ringkasan Evaluasi----")

ringkasan_terminal = pd.DataFrame([
    {
        "Skenario": "70:30",
        "Rules Diuji": jumlah_rules,
        "Rules Konsisten": jumlah_konsisten,
        "Rules Tidak Konsisten": jumlah_tidak_konsisten,
        "Strong Pattern": strong,
        "Moderate Pattern": moderate,
        "Weak Pattern": weak,
        "Rata-rata Support": round(rata_support, 4),
        "Rata-rata Confidence": round(rata_confidence, 4),
        "Rata-rata Lift": round(rata_lift, 4)
    }
])

print(
    tabulate(
        ringkasan_terminal,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)

# KESIMPULAN EVALUASI
print("\n=== KESIMPULAN EVALUASI ===")
print(
    f"Berdasarkan hasil pengujian skenario 70:30, terdapat {jumlah_rules} rules yang diuji. "
    f"Dari jumlah tersebut, {jumlah_konsisten} rules dinyatakan konsisten dan "
    f"{jumlah_tidak_konsisten} rules tidak konsisten. Hasil pengujian menghasilkan "
    f"{strong} Strong Pattern, {moderate} Moderate Pattern, dan {weak} Weak Pattern. "
    f"Rata-rata support yang diperoleh sebesar {rata_support:.4f}, "
    f"rata-rata confidence sebesar {rata_confidence:.4f}, "
    f"dan rata-rata lift sebesar {rata_lift:.4f}."
)