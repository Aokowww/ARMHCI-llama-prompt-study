import os
import zipfile
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs('output', exist_ok=True)

# Load data from the ZIP file
zip_path = 'results.zip'
data_frames = []
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Models present
    models = ['model_1', 'model_2']
    for model in models:
        acc_path = f'results/{model}/accuracy_{model}/accuracy_{model}.csv'
        rep_path = f'results/{model}/reproducibility_{model}/reproducibility_{model}.csv'
        acc_df = pd.read_csv(zip_ref.open(acc_path))
        rep_df = pd.read_csv(zip_ref.open(rep_path))
        # Label data by model
        acc_df['model'] = model
        rep_df['model'] = model
        # Merge accuracy and reproducibility on prompt, shot, format, model
        merged = pd.merge(acc_df, rep_df, on=['prompt_id','shot','format','model'])
        data_frames.append(merged)

# Combine data for all models
data = pd.concat(data_frames, ignore_index=True)

# Convert columns to appropriate types
data['accuracy'] = data['accuracy'].replace('unknown', 0).astype(float)
data['reproducibility'] = data['reproducibility'].astype(float)
data['model'] = data['model'].astype('category')
data['format'] = data['format'].astype('category')
data['shot'] = data['shot'].astype('category')

# Compute mean accuracy and consistency per condition (model, format, shot)
mean_results = (
    data.groupby(['model','format','shot'], observed=False)
        .agg(mean_accuracy=('accuracy','mean'), mean_consistency=('reproducibility','mean'))
        .reset_index()
)
mean_results.to_csv('output/mean_results.csv', index=False)

# Perform three-way ANOVA for accuracy
model_acc = smf.ols('accuracy ~ C(model) * C(format) * C(shot)', data=data).fit()
anova_acc = sm.stats.anova_lm(model_acc, typ=2).reset_index()
anova_acc.rename(columns={'index':'Effect'}, inplace=True)
anova_acc.to_csv('output/anova_accuracy.csv', index=False)

# Perform three-way ANOVA for consistency (reproducibility)
model_cons = smf.ols('reproducibility ~ C(model) * C(format) * C(shot)', data=data).fit()
anova_cons = sm.stats.anova_lm(model_cons, typ=2).reset_index()
anova_cons.rename(columns={'index':'Effect'}, inplace=True)
anova_cons.to_csv('output/anova_consistency.csv', index=False)

# Tukey HSD post hoc comparisons for format effects (accuracy)
mc_acc = MultiComparison(data['accuracy'], data['format'])
tukey_acc = mc_acc.tukeyhsd()
tukey_acc_df = pd.DataFrame(data=tukey_acc._results_table.data[1:],
                            columns=tukey_acc._results_table.data[0])
tukey_acc_df.to_csv('output/tukey_accuracy_format.csv', index=False)

# Tukey HSD post hoc comparisons for format effects (consistency)
mc_cons = MultiComparison(data['reproducibility'], data['format'])
tukey_cons = mc_cons.tukeyhsd()
tukey_cons_df = pd.DataFrame(data=tukey_cons._results_table.data[1:],
                             columns=tukey_cons._results_table.data[0])
tukey_cons_df.to_csv('output/tukey_consistency_format.csv', index=False)

# Set up seaborn style for plots
sns.set_theme(style="whitegrid")

# Boxplot: Accuracy by prompt format
plt.figure(figsize=(6,5))
ax = sns.boxplot(x="format", y="accuracy", data=data)
# Add individual data points (jitter)
sns.stripplot(x="format", y="accuracy", data=data, color="gray", alpha=0.5, jitter=True, ax=ax)
ax.set_xlabel("Prompt Format")
ax.set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig('output/accuracy_by_format.png', dpi=300)
plt.close()

# Line plot: Consistency vs. number of shots per model
mean_consistency = data.groupby(['model','shot'], observed=False)['reproducibility'].mean().reset_index()
plt.figure(figsize=(6,5))
ax2 = sns.lineplot(x='shot', y='reproducibility', hue='model', marker="o", data=mean_consistency)
ax2.set_xlabel("Number of Shots")
ax2.set_ylabel("Mean Consistency")
ax2.set_xticks([0,1,3,5])
ax2.set_xticklabels([0,1,3,5])
plt.legend(title="Model")
plt.tight_layout()
plt.savefig('output/consistency_vs_shots.png', dpi=300)
plt.close()
