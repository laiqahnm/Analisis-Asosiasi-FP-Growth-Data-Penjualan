import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("dataset", exist_ok=True)

df = pd.read_excel("output/dataset_bersih_baru.xlsx")

print("=== DATASET AWAL ===")
print("Jumlah baris data:", len(df))
print("Jumlah transaksi:", df["no_transaksi"].nunique())

# SPLIT DATA BERDASARKAN NO TRANSAKSI

transaksi_unik = df["no_transaksi"].drop_duplicates()

train_transaksi, test_transaksi = train_test_split(
    transaksi_unik,
    test_size=0.4,      
    random_state=42
)

df_training = df[df["no_transaksi"].isin(train_transaksi)].copy()
df_testing = df[df["no_transaksi"].isin(test_transaksi)].copy()

df_training.to_excel("dataset/Dataset Baru/data_training6040.xlsx", index=False)
df_testing.to_excel("dataset/Dataset Baru/data_testing6040.xlsx", index=False)

print("\n=== HASIL SPLIT DATA ===")
print("Jumlah baris training:", len(df_training))
print("Jumlah baris testing:", len(df_testing))

print("Jumlah transaksi training:", df_training["no_transaksi"].nunique())
print("Jumlah transaksi testing:", df_testing["no_transaksi"].nunique())

print("\n FILE TERSIMPAN")