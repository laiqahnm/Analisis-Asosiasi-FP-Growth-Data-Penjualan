import pandas as pd
import glob
import os

folder_path = "dataset/Dataset Baru"

# Ambil semua file Excel
files = glob.glob(os.path.join(folder_path, "*.xls")) + \
        glob.glob(os.path.join(folder_path, "*.xlsx"))

print("---PROSES GABUNG DATA---")
print("Folder yang dibaca:", folder_path)
print("Jumlah file ditemukan:", len(files))
print("File ditemukan:", files)

# Cek apakah file ditemukan
if len(files) == 0:
    print("\nERROR: Tidak ada file Excel ditemukan.")
    print("Pastikan file ada di folder: dataset/Dataset Baru")
    exit()

df_list = []

# read file
for file in files:
    print(f"\nMembaca file: {file}")

    try:
        df = pd.read_excel(file)

        # Rapikan nama kolom
        df.columns = df.columns.str.strip()

        # Tambahkan kolom sumber file
        df["source_file"] = os.path.basename(file)

        df_list.append(df)

        print(f"Berhasil dibaca: {df.shape[0]} baris, {df.shape[1]} kolom")

    except Exception as e:
        print(f"Gagal membaca file {file}")
        print("Error:", e)

# GABUNGKAN DATA
if len(df_list) == 0:
    print("\nERROR: Tidak ada data yang berhasil dibaca.")
    exit()

df_all = pd.concat(df_list, ignore_index=True)

print("\n=== DATA SEBELUM HAPUS DUPLIKAT ===")
print(df_all.shape)

# Hapus duplikat (transaksi harus tercatat satu kali)
df_all = df_all.drop_duplicates()

print("\n---DATA SETELAH HAPUS DUPLIKAT---")
print(df_all.shape)

print("\n---SAMPLE DATA---")
print(df_all.head())

print("\n---NAMA KOLOM---")
print(df_all.columns)

print("\n---DATA KOSONG---")
print(df_all.isnull().sum())

# save
os.makedirs("output", exist_ok=True)

output_path = "output/dataset_gabungan_baru.xlsx"
df_all.to_excel(output_path, index=False)

print("\n---finish---")
print(f"Jumlah data akhir: {df_all.shape}")
print(f"Disimpan di: {output_path}")