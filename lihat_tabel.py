# %%
import pandas as pd
from IPython.display import display

# %%
df = pd.read_excel("output/dataset_gabungan.xlsx")

kolom_tampil = [
    "No Transaksi",
    "Tanggal",
    "Waktu",
    "Operator",
    "Metode Pembayaran",
    "Detail Produk",
    "Banyak Penjualan",
    "Penjualan Bersih"
]

display(df[kolom_tampil].head())

df.info()
# %%
df
# %%

