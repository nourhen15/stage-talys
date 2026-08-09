"""
Découpe creditcard.csv en deux lots :
- initial_data.csv (80%) : utilisé pour le premier entraînement
- new_data.csv (20%) : simule l'arrivée de nouvelles données (pour déclencher le retraining)

On mélange les lignes avant de découper (shuffle) pour que les deux lots
contiennent un mélange représentatif de fraudes/transactions normales.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Charger le dataset complet
df = pd.read_csv("data/raw/creditcard.csv")

print(f"Nombre total de transactions : {len(df)}")
print(f"Nombre de fraudes : {df['Class'].sum()}")

# 2. Découper en 80% (initial) / 20% (nouveau)
# stratify=df["Class"] garantit que la proportion de fraudes est
# respectée dans les deux lots (sinon on risque d'avoir 0 fraude dans un lot)
initial_data, new_data = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Class"]
)

# 3. Sauvegarder les deux lots
initial_data.to_csv("data/raw/initial_data.csv", index=False)
new_data.to_csv("data/raw/new_data.csv", index=False)

print(f"\ninitial_data.csv : {len(initial_data)} lignes, {initial_data['Class'].sum()} fraudes")
print(f"new_data.csv : {len(new_data)} lignes, {new_data['Class'].sum()} fraudes")
print("\nDécoupage terminé.")