import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_excel("output/dataset_bersih.xlsx")

print("=== DATA AWAL ===")
print(df.head())

# BUAT TOKEN ITEM

df["item_produk"] = "produk_" + df["detail_produk"].astype(str)
df["item_operator"] = "operator_" + df["operator"].astype(str)
#df["item_metode"] = "metode_" + df["metode_pembayaran"].astype(str)
df["item_waktu"] = "waktu_" + df["kategori_waktu"].astype(str)

# BENTUK ITEMSET PER TRANSAKSI

basket = df.groupby("no_transaksi").apply(
    lambda x: list(dict.fromkeys(
        x["item_produk"].tolist() +
        x["item_operator"].dropna().tolist() +
        #x["item_metode"].dropna().tolist() +
        x["item_waktu"].dropna().tolist()
    ))
).reset_index(name="itemset")

print("\nJumlah transaksi:", len(basket))
print(basket.head())

# VALIDASI ITEMSET

basket["jumlah_item"] = basket["itemset"].apply(len)

print("\n=== STATISTIK JUMLAH ITEM ===")
print(basket["jumlah_item"].describe())

# TRANSFORM KE ONE HOT (FP-GROWTH)


te = TransactionEncoder()
te_data = te.fit(basket["itemset"]).transform(basket["itemset"])

df_fp = pd.DataFrame(te_data, columns=te.columns_)

print("\n=== DATA SIAP FP-GROWTH ===")
print(df_fp.head())

# save hasil

basket.to_excel("output/dataset_transaksi_itemset.xlsx", index=False)
df_fp.to_excel("output/dataset_fp_growth.xlsx", index=False)

print("\n=== DATA BERHASIL DISIMPAN ===")