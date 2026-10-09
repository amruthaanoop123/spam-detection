import streamlit as st
import pandas as pd
import joblib
import re
from email.utils import parseaddr

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_email_model.pkl")
    preprocessor = joblib.load("spam_email_preprocessor.pkl")
    return model, preprocessor

st.title("📧 Spam Email Detection")
st.write("Check whether an email is likely to be spam or legitimate.")

try:
    model, preprocessor = load_artifacts()
except Exception:
    st.error(
        "Model files not found. Keep spam_email_model.pkl and "
        "spam_email_preprocessor.pkl in the same folder as app.py."
    )
    st.stop()

subject = st.text_input("Email subject")
sender = st.text_input("Sender email address")
message = st.text_area("Email message", height=220)

if st.button("Check Email", type="primary"):
    if not subject.strip() and not message.strip():
        st.warning("Enter an email subject or message first.")
    else:
        _, address = parseaddr(sender)
        address = address.lower().strip()
        domain = address.rsplit("@", 1)[1] if "@" in address else "unknown"

        email_text = ("Subject: " + subject + " " + message)[:10000]

        input_data = pd.DataFrame([{
            "email_text": email_text,
            "sender_domain": domain,
            "message_length": len(message),
            "num_links": len(re.findall(r"https?://|www\.", message)),
            "has_html": int(bool(re.search(
                r"<html|<body|<div|<a\s",
                message,
                flags=re.IGNORECASE
            )))
        }])

        processed = preprocessor.transform(input_data)
        prediction = int(model.predict(processed)[0])
        spam_probability = float(model.predict_proba(processed)[0][1])

        st.metric("Estimated spam probability", f"{spam_probability:.1%}")

        if prediction == 1:
            st.error("⚠️ This email is classified as SPAM.")
        else:
            st.success("✅ This email is classified as HAM (LEGITIMATE).")

        st.caption(
            "This is a machine-learning estimate, not a guarantee. "
            "Review suspicious emails carefully."
        )
