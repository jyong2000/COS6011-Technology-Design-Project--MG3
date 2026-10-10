import streamlit as st
from model_loader import predict_emotion
from evaluation_dashboard import show_dashboard
from methodology import show_methodology

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Emotion Classification", page_icon="📊", layout="wide"
)

# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("Financial Emotion Classification")
st.caption(
    "COS60011 Technology Design Project | " "Transfer Learning and Domain Adaptation"
)

st.divider()

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

classifier_tab, comparison_tab, about_tab = st.tabs(
    ["Emotion Classifier", "Model Comparison", "About / Methodology"]
)

# --------------------------------------------------
# PAGE 1: EMOTION CLASSIFIER
# --------------------------------------------------

with classifier_tab:

    st.header("Financial Emotion Classifier")

    st.write(
        "Enter financial-related text to classify its emotion using the "
        "Dataset Weighting BERT model selected for this demonstration."
    )

    st.caption(
        "Selected BERT strategy: Dataset Weighting | Highest test Macro-F1 among four BERT strategies"
    )

    user_text = st.text_area(
        "Enter Financial Text",
        placeholder=(
            "Example: The company reported strong earnings "
            "and its share price increased significantly."
        ),
        height=150,
    )

    if st.button("Analyse Emotion", type="primary"):

        if not user_text.strip():
            st.warning("Please enter financial text.")

        else:
            try:
                with st.spinner("Analysing financial emotion..."):
                    result = predict_emotion(user_text)

                st.success("Analysis completed!")

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Predicted Emotion", result["emotion"].capitalize())

                with col2:
                    st.metric("Model Confidence", f"{result['confidence']:.2%}")

                st.subheader("Emotion Probabilities")

                for emotion, probability in sorted(
                    result["probabilities"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    st.write(f"**{emotion.capitalize()}**: {probability * 100:.2f}%")

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# PAGE 2: MODEL COMPARISON
# --------------------------------------------------

with comparison_tab:
    show_dashboard()

# --------------------------------------------------
# PAGE 3: ABOUT / METHODOLOGY
# --------------------------------------------------

with about_tab:
    show_methodology()
