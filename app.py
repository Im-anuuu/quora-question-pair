import joblib
from flask import Flask, render_template, request
from features import build_features

app = Flask(__name__)
model = joblib.load("model.joblib")
tfidf = joblib.load("tfidf.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        q1 = request.form["q1"]
        q2 = request.form["q2"]
        X = build_features([q1], [q2], tfidf)
        prob = model.predict_proba(X)[0][1]
        result = {"duplicate": prob >= 0.5, "prob": round(prob * 100, 1)}
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)