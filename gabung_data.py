import pandas as pd
import glob
import os

#GABUNGKAN SEMUA DATA
folder_path = "dataset"
files = glob.glob(os.path.join(folder_path, "*.xls")) + \
        glob.glob(os.path.join(folder_path, "*.xlsx"))

df_list = []

print("=== PROSES GABUNG DATA ===")

for file in files:
    print(f"Membaca file: {file}")
    
    try:
        df = pd.read_excel(file)
        df['source_file'] = os.path.basename(file)  # optional tracking
        df_list.append(df)
    except Exception as e:
        print(f"Error di file {file}: {e}")

df_all = pd.concat(df_list, ignore_index=True)
print("\n=== SAMPLE DATA ===")
print(df_all.head())

print("\n=== INFO DATA ===")
print(df_all.info())

print("\n=== DATA KOSONG ===")
print(df_all.isnull().sum())

print("\n=== NAMA KOLOM ===")
print(df_all.columns)

os.makedirs("output", exist_ok=True)

output_path = "output/dataset_gabungan.xlsx"
df_all.to_excel(output_path, index=False)

print("\n=== SELESAI ===")
print(f"Jumlah data: {df_all.shape}")
print(f"Disimpan di: {output_path}")

print(df_all.head())