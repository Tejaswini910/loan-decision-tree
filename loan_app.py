from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


DATA_PATH = Path(__file__).resolve().with_name("loan_data 1.csv")
TARGET = "not.fully.paid"
REQUIRED_COLUMNS = {
    "credit.policy",
    "purpose",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
    TARGET,
}

st.set_page_config(page_title="Loan Decision Tree", page_icon="L", layout="wide")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing_columns))}")
    if data[TARGET].isna().any() or not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0 and 1 values without missing entries.")
    return data


st.title("Loan Repayment Decision Tree")
st.caption("Explore historical loan outcomes and evaluate a decision-tree classifier.")
st.warning(
    "Educational analysis only. This model is not validated for lending decisions and should not be used to approve or deny credit."
)

try:
    loan_data = load_data(DATA_PATH)
except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Could not load the loan dataset: {error}")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    test_size = st.slider("Test-set share", min_value=0.1, max_value=0.4, value=0.25, step=0.05)
    depth_choice = st.selectbox("Maximum tree depth", ["Unlimited", 2, 3, 4, 5, 6, 8, 10, 12], index=4)
    min_samples_leaf = st.slider("Minimum samples per leaf", min_value=1, max_value=100, value=20)
    criterion = st.selectbox("Split criterion", ["gini", "entropy", "log_loss"])
    balance_classes = st.checkbox("Balance class weights", value=True)

max_depth = None if depth_choice == "Unlimited" else int(depth_choice)
features = loan_data.drop(columns=TARGET)
target = loan_data[TARGET].astype(int)
categorical_columns = features.select_dtypes(include=["object", "category"]).columns.tolist()
preprocessor = ColumnTransformer(
    [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
classifier = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    class_weight="balanced" if balance_classes else None,
    random_state=42,
)
model = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=test_size,
    random_state=42,
    stratify=target,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

positive_rate = target.mean() * 100
metric_columns = st.columns(5)
metric_columns[0].metric("Loan records", f"{len(loan_data):,}")
metric_columns[1].metric("Not fully paid", f"{positive_rate:.1f}%")
metric_columns[2].metric("Test accuracy", f"{accuracy_score(y_test, predictions):.1%}")
metric_columns[3].metric("Balanced accuracy", f"{balanced_accuracy_score(y_test, predictions):.1%}")
metric_columns[4].metric("Recall: not fully paid", f"{recall_score(y_test, predictions, zero_division=0):.1%}")

performance_tab, tree_tab, data_tab = st.tabs(["Performance", "Decision tree", "Dataset"])

with performance_tab:
    left_column, right_column = st.columns([1, 1.5])
    with left_column:
        st.subheader("Test-set confusion matrix")
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        matrix_frame = pd.DataFrame(
            matrix,
            index=["Actual: fully paid", "Actual: not fully paid"],
            columns=["Predicted: fully paid", "Predicted: not fully paid"],
        )
        st.dataframe(matrix_frame, width="stretch")
        st.caption("Rows are actual outcomes; columns are model predictions.")

    with right_column:
        st.subheader("Metrics by outcome")
        metric_frame = pd.DataFrame(
            {
                "Precision": precision_score(y_test, predictions, average=None, labels=[0, 1], zero_division=0),
                "Recall": recall_score(y_test, predictions, average=None, labels=[0, 1], zero_division=0),
                "F1 score": f1_score(y_test, predictions, average=None, labels=[0, 1], zero_division=0),
            },
            index=["Fully paid (0)", "Not fully paid (1)"],
        )
        st.dataframe(metric_frame.style.format("{:.1%}"), width="stretch")

    st.subheader("Most influential features")
    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_tree = model.named_steps["classifier"]
    feature_importance = pd.DataFrame(
        {
            "Feature": fitted_preprocessor.get_feature_names_out(),
            "Importance": fitted_tree.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    st.bar_chart(feature_importance.head(12).set_index("Feature"), height=360)

with tree_tab:
    st.subheader("Tree structure (first three levels)")
    st.caption("Use the controls in the sidebar to change tree depth and leaf size.")
    figure, axis = plt.subplots(figsize=(22, 10))
    plot_tree(
        fitted_tree,
        feature_names=fitted_preprocessor.get_feature_names_out(),
        class_names=["Fully paid", "Not fully paid"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=axis,
    )
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with data_tab:
    st.subheader("Loan data")
    st.dataframe(loan_data, width="stretch", hide_index=True)
    st.download_button(
        "Download test-set predictions",
        data=pd.DataFrame(
            {
                "actual_not_fully_paid": y_test,
                "predicted_not_fully_paid": predictions,
            }
        ).to_csv(index=False).encode("utf-8"),
        file_name="loan_test_predictions.csv",
        mime="text/csv",
    )