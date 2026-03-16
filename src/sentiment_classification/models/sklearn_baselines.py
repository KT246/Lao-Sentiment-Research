import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, hinge_loss, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


DEFAULT_TFIDF_CONFIG = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_features": 50000,
    "sublinear_tf": True,
    "lowercase": False,
}


def build_text_vectorizer(tfidf_config=None):
    merged_tfidf_config = dict(DEFAULT_TFIDF_CONFIG)
    if tfidf_config:
        merged_tfidf_config.update(tfidf_config)

    return TfidfVectorizer(**merged_tfidf_config)


def _sgd_alpha_from_c(c_value: float):
    safe_c = max(float(c_value), 1e-6)
    return 1.0 / (safe_c * 10000.0)


def build_baseline_estimator(
    model_key: str,
    seed: int = 42,
    class_weight=None,
    logreg_c: float = 1.0,
    svm_c: float = 1.0,
    decision_tree_max_depth: int = 40,
    decision_tree_min_samples_leaf: int = 2,
):
    if model_key == "logistic-regression":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=_sgd_alpha_from_c(logreg_c),
            class_weight=class_weight,
            shuffle=False,
            random_state=seed,
        )

    if model_key == "decision-tree":
        return DecisionTreeClassifier(
            class_weight=class_weight,
            max_depth=decision_tree_max_depth,
            min_samples_leaf=decision_tree_min_samples_leaf,
            max_features="sqrt",
            random_state=seed,
        )

    if model_key == "svm":
        return SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=_sgd_alpha_from_c(svm_c),
            class_weight=class_weight,
            shuffle=False,
            random_state=seed,
        )

    raise ValueError(f"Unsupported sklearn baseline model_key: {model_key}")


def build_sklearn_text_pipeline(vectorizer, classifier):
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def predict_with_confidence(model, inputs):
    predicted_labels = model.predict(inputs)
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model

    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba(inputs)
        confidence_scores = probabilities.max(axis=1)
        confidence_method = "predict_proba"
    elif hasattr(classifier, "decision_function"):
        decision_scores = model.decision_function(inputs)
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


def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 6),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 6),
    }


def compute_eval_loss(model_key, classifier, inputs, labels):
    if model_key == "svm":
        decision_scores = classifier.decision_function(inputs)
        return round(float(hinge_loss(labels, decision_scores)), 6)

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(inputs)
        return round(float(log_loss(labels, probabilities, labels=sorted(np.unique(labels)))), 6)

    if hasattr(classifier, "decision_function"):
        decision_scores = classifier.decision_function(inputs)
        decision_scores = np.asarray(decision_scores)
        if decision_scores.ndim == 1:
            probabilities_pos = 1.0 / (1.0 + np.exp(-decision_scores))
            probabilities = np.column_stack([1.0 - probabilities_pos, probabilities_pos])
        else:
            stable_scores = decision_scores - decision_scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(stable_scores)
            probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return round(float(log_loss(labels, probabilities, labels=sorted(np.unique(labels)))), 6)

    predictions = classifier.predict(inputs)
    return round(float(np.mean(predictions != labels)), 6)


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
