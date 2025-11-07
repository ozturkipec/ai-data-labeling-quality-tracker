import streamlit as st
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Data Labeling Quality Tracker", layout="wide")

# --- HEADER (clean + professional) ---
st.markdown(
    """
    <style>
        .banner {
            background: linear-gradient(90deg, #004aad, #007bff);
            padding: 1.5em;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2em;
        }
        .banner h1 {
            font-size: 2em;
            font-weight: 700;
            margin: 0;
        }
        .banner p {
            font-size: 1.1em;
            margin-top: 0.3em;
            color: #e5e5e5;
        }
    </style>
    <div class="banner">
        <h1>AI Data Labeling Quality Tracker</h1>
        <p>Evaluate AI model predictions, labeling accuracy, and edge cases interactively.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="File should include true and predicted label columns.",
)

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # --- COLUMN SELECTION ---
    st.sidebar.subheader("Select Columns")
    cols = df.columns.tolist()
    true_col = st.sidebar.selectbox("True label column", cols)
    pred_col = st.sidebar.selectbox("Predicted label column", cols)

    if true_col and pred_col:
        y_true = df[true_col]
        y_pred = df[pred_col]

        # Drop missing rows
        mask = y_true.notna() & y_pred.notna()
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            st.error("No valid rows after removing missing values. Check your data.")
        else:
            # --- METRICS ---
            labels = sorted(y_true.unique())
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, zero_division=0
            )

            st.subheader("📊 Overall Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("Macro Precision", f"{precision.mean():.3f}")
            col3.metric("Macro Recall", f"{recall.mean():.3f}")

            # --- PER-LABEL METRICS ---
            metrics_df = pd.DataFrame({
                "Label": labels,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Support": support
            })
            st.subheader("🔍 Per-label Metrics")
            st.dataframe(metrics_df.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}"
            }))

            # --- CONFUSION MATRIX ---
            st.subheader("🧮 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index=[f"True: {l}" for l in labels],
                columns=[f"Pred: {l}" for l in labels]
            )
            st.dataframe(cm_df)

            # --- ERROR EXPLORER ---
            st.subheader("🚨 Error Explorer (False Positives / False Negatives)")
            selected_label = st.selectbox(
                "Select label to inspect",
                labels,
                help="Choose a label to explore false positives and negatives."
            )

            df_valid = df[mask].copy()
            df_valid["__true__"] = y_true.values
            df_valid["__pred__"] = y_pred.values

            # False Positives: predicted = label, true ≠ label
            fp_rows = df_valid[
                (df_valid["__pred__"] == selected_label)
                & (df_valid["__true__"] != selected_label)
            ]

            # False Negatives: true = label, predicted ≠ label
            fn_rows = df_valid[
                (df_valid["__true__"] == selected_label)
                & (df_valid["__pred__"] != selected_label)
            ]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**False Positives for label `{selected_label}`**")
                if len(fp_rows) == 0:
                    st.info("No false positives for this label ✅")
                else:
                    st.dataframe(fp_rows.drop(columns=["__true__", "__pred__"]).head(100))

            with c2:
                st.markdown(f"**False Negatives for label `{selected_label}`**")
                if len(fn_rows) == 0:
                    st.info("No false negatives for this label ✅")
                else:
                    st.dataframe(fn_rows.drop(columns=["__true__", "__pred__"]).head(100))

else:
    st.info("👈 Upload a CSV file in the sidebar to get started.")
    st.write(
        "Example columns:\n"
        "- `true_label`\n"
        "- `predicted_label`\n\n"
        "You can also include extra columns like timestamps, IDs, or notes. "
        "They’ll appear when inspecting false positives and false negatives."
    )