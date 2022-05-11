import pandas as pd
df = pd.read_csv("hetero-star-node/ques_f1_contrast.csv")
df.head(10)
print(df.groupby('cat').mean())
df_2= df[df["cat"]!="text"]
df_2["f1"].mean()
print(df_2["f1"].mean())