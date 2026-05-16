import pandas as pd

df = pd.read_excel("output/dataset_gabungan_baru.xlsx")

# select column
select_column = [
    "No Transaksi",
    "Tanggal",
    "Waktu",
    "Operator",
    #"Metode Pembayaran",
    "Detail Produk",
    "Penjualan Bersih"
]

df = df[select_column].copy()

print("=== SELECTED COLUMN ===")
print(df.columns.tolist())

# rename kolom
df = df.rename(columns={
    "No Transaksi": "no_transaksi",
    "Tanggal": "tanggal",
    "Waktu": "waktu",
    "Operator": "operator",
    #"Metode Pembayaran": "metode_pembayaran",
    "Detail Produk": "detail_produk",
    "Penjualan Bersih": "penjualan_bersih"
})

print("=== KOLOM SETELAH RENAME ===")
print(df.columns.tolist())

# Hapus operator yang penjualannya sedikit
df["operator"] = df["operator"].astype(str).str.strip().str.lower()

operator_dihapus = ["devi", "dina", "owner"]

df = df[~df["operator"].isin(operator_dihapus)]

print("Jumlah data setelah operator tertentu dihapus:")
print(df["operator"].value_counts())

# hapus item non-produk
print("\n=== FILTER PRODUK NON-SKINCARE ===")
print("Jumlah data sebelum filter non-produk:", df.shape[0])

df["detail_produk"] = df["detail_produk"].astype(str).str.strip()

kata_non_produk = ["kardus", "lakban", "brosur", "clutch", "pouch", "boxi", "paper bag", "paperbag", "bubble wrap", "tas", "plastik", "tenteng", "cermin", "ongkir"]
pattern = "|".join(kata_non_produk)

df = df[~df["detail_produk"].str.contains(pattern, case=False, na=False)]

print("Jumlah data sesudah filter non-produk:", df.shape[0])

# missing value
print("\n=== CEK MISSING VALUE ===")
print(df.isnull().sum())

# duplikat
print("\n=== CEK DUPLIKAT ===")
print("Jumlah duplikat:", df.duplicated().sum())

# hapus duplikat
print("\n=== HAPUS DUPLIKAT ===")
print("Jumlah data sebelum hapus duplikat:", df.shape[0])

df = df.drop_duplicates()

print("Jumlah data sesudah hapus duplikat:", df.shape[0])

# rapikan isi kolom teks
kolom_teks = [
    "no_transaksi",
    "operator",
    #"metode_pembayaran",
    "detail_produk"
]

for col in kolom_teks:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip() #spasi dpn blkg
        df[col] = df[col].str.replace(r"\s+", " ", regex=True) #spasi ganda

# standarisasi format
if "no_transaksi" in df.columns:
    df["no_transaksi"] = df["no_transaksi"].str.upper()

if "operator" in df.columns:
    df["operator"] = df["operator"].str.title()

#if "metode_pembayaran" in df.columns:
    #df["metode_pembayaran"] = df["metode_pembayaran"].str.lower()

    #df["metode_pembayaran"] = df["metode_pembayaran"].replace({
        #"shopee pay": "shopeepay"
    #})

if "detail_produk" in df.columns:
    df["detail_produk"] = df["detail_produk"].str.lower()
    df["detail_produk"] = df["detail_produk"].str.replace(r"\bpkt\b", "paket", regex=True)


# ubah tipe data
if "tanggal" in df.columns:
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce").dt.date

if "penjualan_bersih" in df.columns:
    df["penjualan_bersih"] = pd.to_numeric(df["penjualan_bersih"], errors="coerce")

print("\n=== INFO DATA SETELAH UBAH TIPE ===")
print(df.info())

# ubah kategori waktu

df["waktu_parsed"] = pd.to_datetime(df["waktu"], errors="coerce")
df["jam"] = df["waktu_parsed"].dt.hour

def kategori_waktu(jam):
    if pd.isna(jam):
        return pd.NA
    elif 0 <= jam < 12:
        return "Pagi"
    elif 12 <= jam < 18:
        return "Siang"
    else:
        return "Malam"

df["kategori_waktu"] = df["jam"].apply(kategori_waktu)
df["waktu"] = df["waktu_parsed"].dt.strftime("%H:%M")
df = df.drop(columns=["waktu_parsed", "jam"])

print("\n=== HASIL WAKTU DAN KATEGORI ===")
print(df[["waktu", "kategori_waktu"]].head(10))

# cleansing hasil refund
print("\n=== FILTER HASIL REFUND  ===")
print("Jumlah data sebelum filter minus:", df.shape[0])


df = df[df["penjualan_bersih"] > 0]

print("Jumlah data sesudah filter minus:", df.shape[0])

# analisis deskriptif produk
print("\n=== JUMLAH PER PRODUK TERJUAL ===")
produk_rekap = df["detail_produk"].value_counts().reset_index()
produk_rekap.columns = ["nama_produk", "jumlah_terjual"]
print(produk_rekap.head())

produk_rekap = produk_rekap.sort_values(by="jumlah_terjual", ascending=False)
produk_rekap.to_excel("output/rekap_produk.xlsx", index=False)

# analisis distribusi waktu
print("\n=== DISTRIBUSI PENJUALAN BERDASARKAN WAKTU ===")

waktu_dist = (df["kategori_waktu"].value_counts(normalize=True) * 100).reset_index()
waktu_dist.columns = ["kategori_waktu", "persentase"]
print(waktu_dist)

# hitung persentase penjualan per operator
print("\n=== ANALISIS OPERATOR BERDASARKAN WAKTU ===")

operator_waktu = df.groupby(["operator", "kategori_waktu"]).size().reset_index(name="jumlah")
operator_waktu["persentase"] = operator_waktu.groupby("operator")["jumlah"].transform(lambda x: x / x.sum() * 100)
operator_waktu = operator_waktu.sort_values(by="jumlah", ascending=False).reset_index(drop=True) 

print(operator_waktu.sort_values(by="jumlah", ascending=False).head(20))

# validasi data unik
print("\n=== RINGKASAN DATA ===")
print("Jumlah transaksi unik:", df["no_transaksi"].nunique())
print("Jumlah produk unik:", df["detail_produk"].nunique())
print("Jumlah operator unik:", df["operator"].nunique())
#print("Jumlah metode pembayaran unik:", df["metode_pembayaran"].nunique())

# save data
df.to_excel("output/dataset_bersih_baru.xlsx", index=False)