# -----------------------------------------------------------------------------
# Generates four comparative boxplots for two models showing:
#   1. Accuracy by Prompt Format
#   2. Accuracy by Number of Shots
#   3. Reproducibility by Prompt Format
#   4. Reproducibility by Number of Shots
import os
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Data source configuration
# -----------------------------------------------------------------------------
zip_path = 'results.zip'
models = ['model_1', 'model_2']

# -----------------------------------------------------------------------------
# 2. Load and merge data
# -----------------------------------------------------------------------------
data_frames = []
if zip_path and os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        for model in models:
            acc = pd.read_csv(z.open(f'results/{model}/accuracy_{model}/accuracy_{model}.csv'))
            rep = pd.read_csv(z.open(f'results/{model}/reproducibility_{model}/reproducibility_{model}.csv'))
            acc['model'] = model
            rep['model'] = model
            data_frames.append(pd.merge(acc, rep, on=['prompt_id','shot','format','model']))
else:
    for model in models:
        acc = pd.read_csv(f'results/{model}/accuracy_{model}/accuracy_{model}.csv')
        rep = pd.read_csv(f'results/{model}/reproducibility_{model}/reproducibility_{model}.csv')
        acc['model'] = model
        rep['model'] = model
        data_frames.append(pd.merge(acc, rep, on=['prompt_id','shot','format','model']))

df = pd.concat(data_frames, ignore_index=True)

# -----------------------------------------------------------------------------
# 3. Data preprocessing
# -----------------------------------------------------------------------------
df['accuracy']        = pd.to_numeric(df['accuracy'], errors='coerce')
df['reproducibility'] = pd.to_numeric(df['reproducibility'], errors='coerce')
df['shot']            = df['shot'].astype(str)
df['format']          = df['format'].astype(str)

# -----------------------------------------------------------------------------
# 4. Seaborn configuration
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
format_order = ['text','markdown','yaml','json']
shot_order   = ['0','1','3','5']

# Model-specific colors
box_palette     = {'model_1':'#1f77b4','model_2':'#ff7f0e'}
scatter_palette = {'model_1':'gray','model_2':'gray'}

# -----------------------------------------------------------------------------
# 5. Plotting helper
# -----------------------------------------------------------------------------
def draw_box_by_model(x, y, order, xlabel, ylabel, title, outfile):
    """
    Draws boxplot split by model with:
      - Prominent median lines
      - Overlayed gray points
      - Legend outside plot for clear labeling
    """
    plt.figure(figsize=(8,5))
    ax = sns.boxplot(
        x=x, y=y, hue='model', data=df,
        order=order, palette=box_palette,
        showfliers=True, fliersize=5,
        medianprops={'color':'black','linewidth':3},
        whiskerprops={'linewidth':1.5},
        capprops={'linewidth':1.5},
        boxprops={'linewidth':1.5}
    )
    # Overlay jittered gray points
    sns.stripplot(
        x=x, y=y, hue='model', data=df,
        order=order, palette=scatter_palette,
        dodge=True, jitter=True, alpha=0.3, size=5,
        ax=ax
    )
    # Move legend outside to the right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:len(models)], labels[:len(models)],
        title='Model',
        bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, outfile), dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# 6. Generate all four plots
# -----------------------------------------------------------------------------
draw_box_by_model(
    x='format', y='accuracy', order=format_order,
    xlabel='Prompt Format', ylabel='Accuracy',
    title='Accuracy by Prompt Format and Model',
    outfile='accuracy_by_format_model_updated.png'
)

draw_box_by_model(
    x='shot', y='accuracy', order=shot_order,
    xlabel='Number of Shots', ylabel='Accuracy',
    title='Accuracy by Number of Shots and Model',
    outfile='accuracy_by_shots_model_updated.png'
)

draw_box_by_model(
    x='format', y='reproducibility', order=format_order,
    xlabel='Prompt Format', ylabel='Reproducibility',
    title='Reproducibility by Prompt Format and Model',
    outfile='reproducibility_by_format_model_updated.png'
)

draw_box_by_model(
    x='shot', y='reproducibility', order=shot_order,
    xlabel='Number of Shots', ylabel='Reproducibility',
    title='Reproducibility by Number of Shots and Model',
    outfile='reproducibility_by_shots_model_updated.png'
)

print("Updated plots with emphasized medians and external legend saved to 'output/' directory.")
