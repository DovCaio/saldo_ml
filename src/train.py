import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("../data/raw/train_transactions.csv")
df_test  = pd.read_csv("../data/raw/test_transactions.csv")

frames = [df_train, df_test]

df = pd.concat(frames)

print(df)

X = df["description"].astype(str)
y = df["category"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Treino:", len(X_train))
print("Teste:", len(X_test))
print("\nDistribuição das categorias:")
print(y.value_counts())


model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=100_000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight=None
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)



print("classification report")
print(classification_report(y_test, predictions))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")

plt.tight_layout()
plt.show()

samples = [
    "Uber trip payment",
    "Monthly payment for Netflix",
    "Electricity bill",
    "Supermarket purchase",
    "Hospital consultation",
    "Pix recebido de João",
    "Pagamento via cartão de débito",
    "Transferência recebida"
]

predictions = model.predict(samples)

for description, category in zip(samples, predictions):
    print(f"{description:45} -> {category}")