import pandas as pd


df_answer = pd.read_csv("answer.csv")


results = [f"emotion_results_{i}.csv" for i in range(1, 11)]
dfs = [pd.read_csv(file) for file in results]

num_rows = len(dfs[0])
print(f"number of rows: {num_rows}")

row_match_ratios = []
counter_pairs = 10
#counter_true = 0

for row_idx in range(num_rows):
    prompt_id = int(dfs[0].iloc[row_idx, 0])
    print(f"Prompt ID at row {row_idx+1}: {prompt_id}")
   
    row_outputs = [str(df.iloc[row_idx, -1]).strip().lower() for df in dfs]
    answer_val = str(df_answer.iloc[row_idx, -1]).strip().lower()
    print(f"Answer at row {row_idx+1}: {answer_val}") 
    print(f"Row {row_idx+1} outputs: {[repr(val) for val in row_outputs]}")
   # positive
    if prompt_id in [3, 5]:
        row_outputs = ["positive" if val == "neutral" else val for val in row_outputs]

    counter_true = 0
    counter_unknown = 0
    
    for i in range(len(row_outputs)):
        if row_outputs[i] == 'unknown':
            counter_unknown+= 1

        else:
            if row_outputs[i] == answer_val:
                counter_true += 1
    counter_pairs_final = counter_pairs - counter_unknown
    if counter_pairs_final == 0:
        match_ratio= "unknown"
    else:
        match_ratio = counter_true / counter_pairs_final
    # row_match_ratios.append(round(match_ratio, 2))
    if match_ratio == "unknown":
        row_match_ratios.append(match_ratio)
        print(f"Row {row_idx+1}: Exact Match Ratio = unknown, Matches: {counter_true}/{counter_pairs_final}")
    else:
        row_match_ratios.append(round(match_ratio, 2))
        print(f"Row {row_idx+1}: Exact Match Ratio = {match_ratio:.2f}, Matches: {counter_true}/{counter_pairs_final}")
 

    # print(f"Row {row_idx+1}: Exact Match Ratio = {match_ratio:.2f}, Matches: {counter_true}/{counter_pairs_final}")

df_base = dfs[0].iloc[:, :4].copy()
df_base = df_base.iloc[:, [0, 2, 3]]


df_base['accuracy'] =  row_match_ratios
df_base.to_csv("accuracy_model_1.csv", index=False)

