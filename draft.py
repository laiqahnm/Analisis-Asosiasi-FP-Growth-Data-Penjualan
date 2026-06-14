import pandas as pd
from IPython.display import display

df = pd.read_excel("output/frequent_itemsets.xlsx")

display(df.head(10))