```python
import streamlit as st
import joblib

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_email_nb_model.pkl")
    vectorizer = joblib.load("spam_email_nb_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error("Could not load the Naive Bayes model files.")
    st.exception(e)
    st.stop()

st.title("📧 Spam Email Detection")
st.write(
    "Check whether an email looks like spam or a legitimate message "
    "using a machine-learning model."
)

subject = st.text_input("Email Subject")
sender = st.text_input("Sender Email Address (optional)")
message = st.text_area("Email Message", height=220)

if st.button("Check Email", type="primary"):
    if not subject.strip() and not message.strip():
        st.warning("Please enter an email subject or message.")
    else:
        email_text = ("Subject: " + subject + " " + message)[:10000]
        features = vectorizer.transform([email_text])

        prediction = int(model.predict(features)[0])
        spam_probability = float(
            model.predict_proba(features)[0][1]
        )

        st.metric(
            "Estimated Spam Probability",
            f"{spam_probability:.2%}"
        )

        if prediction == 1:
            st.error("⚠️ This email is classified as SPAM.")
        else:
            st.success("✅ This email is classified as HAM (LEGITIMATE).")

        st.caption(
            "This is a machine-learning estimate, not a guarantee. "
            "Review suspicious emails carefully."
        )
```
