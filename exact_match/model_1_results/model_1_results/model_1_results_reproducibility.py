import pandas as pd

results = [f"emotion_results_{i}.csv" for i in range(1, 11)]
dfs = [pd.read_csv(file) for file in results ]

num_rows = len(dfs[0])
print(f"number of rows: {num_rows}")

row_match_ratios = []
counter_pairs = 45

for row_idx in range(num_rows):
    row_outputs = [str(df.iloc[row_idx, -1]).strip().lower() for df in dfs]
    counter_true = 0
    
    for i in range(len(row_outputs)):
        for j in range(i + 1, len(row_outputs)):
            if row_outputs[i] == row_outputs[j]:
                counter_true += 1

    match_ratio = counter_true / counter_pairs
    row_match_ratios.append(f"{match_ratio:.2f}")  

    print(f"Row {row_idx+1}: Exact Match Ratio = {match_ratio:.2f}, Matches: {counter_true}/45")

df_base = dfs[0].iloc[:, :4].copy()
df_base = df_base.iloc[:, [0, 2, 3]]


df_base['reproducibility'] =  row_match_ratios
df_base.to_csv("reproducibility_model_1.csv", index=False)

