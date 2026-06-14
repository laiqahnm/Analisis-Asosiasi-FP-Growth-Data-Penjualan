# %%
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_excel("output/testing_result_databaru/ringkasan_testing.xlsx")

display(df)
# %%
