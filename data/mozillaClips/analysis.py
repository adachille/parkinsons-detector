import pandas as pd

df = pd.read_csv("validatedFiltered.csv")
print(df.count())
print(df["age"].value_counts())
