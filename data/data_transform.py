
import pandas as pd

original = pd.read_csv(
    "./processed/transactions_pt_br.csv"
)

new_20_more_per_category = pd.read_csv(
    "./raw/new_transactions_pt_br.csv"
)

gemini_generation = pd.read_csv(
    "./raw/gemini_transactions_pt_br.csv"
)

deep_seek_generation = pd.read_csv(
    "./raw/deepseek_pt_br.csv"
)

updated = pd.concat(
    [original, new_20_more_per_category, gemini_generation, deep_seek_generation],
    ignore_index=True
)

# Remove somente descrições exatamente iguais
updated = updated.drop_duplicates(
    subset=["description", "category"]
)

updated.to_csv(
    "./processed/transactions_pt_br_v4.csv",
    index=False
)

print(updated["category"].value_counts())
print("Total:", len(updated))
