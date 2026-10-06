import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from features import build_features, clean

df = pd.read_csv("question.csv").dropna(subset=["question1", "question2"])
df = df.sample(100_000, random_state=42)  # remove this line to use all ~400K rows

# fit TF-IDF on all questions
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
tfidf.fit(pd.concat([df["question1"], df["question2"]]).map(clean))

X = build_features(df["question1"], df["question2"], tfidf)
y = df["is_duplicate"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print("Accuracy:", round(accuracy_score(y_te, pred), 4))
print("F1:", round(f1_score(y_te, pred), 4))
print(classification_report(y_te, pred))

joblib.dump(model, "model.joblib")
joblib.dump(tfidf, "tfidf.joblib")