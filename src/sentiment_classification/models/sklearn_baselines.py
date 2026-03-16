import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


DEFAULT_TFIDF_CONFIG = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_features": 50000,
    "sublinear_tf": True,
    "lowercase": False,
}


def build_sklearn_text_pipeline(
    model_key: str,
    seed: int = 42,
    tfidf_config=None,
    logreg_c: float = 1.0,
    svm_c: float = 1.0,
    decision_tree_max_depth: int = 40,
    decision_tree_min_samples_leaf: int = 2,
):
    merged_tfidf_config = dict(DEFAULT_TFIDF_CONFIG)
    if tfidf_config:
        merged_tfidf_config.update(tfidf_config)

    vectorizer = TfidfVectorizer(**merged_tfidf_config)

    if model_key == "logistic-regression":
        classifier = LogisticRegression(
            C=logreg_c,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
    elif model_key == "decision-tree":
        classifier = DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=decision_tree_max_depth,
            min_samples_leaf=decision_tree_min_samples_leaf,
            max_features="sqrt",
            random_state=seed,
        )
    elif model_key == "svm":
        classifier = LinearSVC(
            C=svm_c,
            class_weight="balanced",
            dual=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported sklearn baseline model_key: {model_key}")

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def predict_with_confidence(pipeline, texts):
    predicted_labels = pipeline.predict(texts)
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "predict_proba"):
        probabilities = pipeline.predict_proba(texts)
        confidence_scores = probabilities.max(axis=1)
        confidence_method = "predict_proba"
    elif hasattr(classifier, "decision_function"):
        decision_scores = pipeline.decision_function(texts)
        decision_scores = np.asarray(decision_scores)
        if decision_scores.ndim == 1:
            confidence_scores = 1.0 / (1.0 + np.exp(-np.abs(decision_scores)))
        else:
            confidence_scores = 1.0 / (1.0 + np.exp(-np.max(np.abs(decision_scores), axis=1)))
        confidence_method = "decision_function_sigmoid"
    else:
        confidence_scores = np.ones(len(predicted_labels), dtype=float)
        confidence_method = "constant_1.0"

    return predicted_labels, confidence_scores, confidence_method


def build_feature_summary(pipeline):
    vectorizer = pipeline.named_steps["tfidf"]
    return {
        "analyzer": vectorizer.analyzer,
        "ngram_range": list(vectorizer.ngram_range),
        "min_df": vectorizer.min_df,
        "max_features": vectorizer.max_features,
        "sublinear_tf": vectorizer.sublinear_tf,
        "lowercase": vectorizer.lowercase,
        "num_features": len(vectorizer.vocabulary_),
    }
