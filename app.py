import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import re
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


st.set_page_config(
    page_title="Clinical Trial Disease Prediction",
    page_icon="🧬",
    layout="wide")

st.sidebar.title("🧬 Clinical Trial NLP")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Disease Analysis",
        "🧠 Disease Prediction",
    ]
)
@st.cache_data
def load_data():
    df = pd.read_csv("processed_clinical_trials.csv")
    return df
@st.cache_resource
def load_models():

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("lr_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return tfidf, model, label_encoder
df = load_data()
tfidf, model, label_encoder = load_models()
if page == "🏠 Home":

    st.title("🧬 Clinical Trial Disease Prediction")

    st.markdown("""
    ### Natural Language Processing Application
    This application analyzes **clinical trial summaries** and predicts the associated disease category using Machine Learning and NLP techniques.""")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trials",f"{len(df):,}")

    with col2:
        if "source_condition_query" in df.columns:
            st.metric("Disease Categories",f"{df['source_condition_query'].nunique():,}")
        else:
            st.metric("Disease Categories", "N/A")

    with col3:
        if "brief_summary" in df.columns:
            st.metric("Clinical Summaries",f"{df['brief_summary'].notna().sum():,}")
        else:
            st.metric("Clinical Summaries", "N/A")

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True)

    st.divider()

    
elif page == "📊 Disease Analysis":

    st.title("📊 Disease Category Analysis")

    if "source_condition_query" not in df.columns:

        st.error(
            "The dataset does not contain the "
            "'source_condition_query' column.")
        st.stop()

    st.subheader("Most Common Disease Categories")

    top_n = st.slider("Number of disease categories",
        min_value=1,max_value=10,value=10)
    

    disease_counts = (
        df["source_condition_query"].value_counts().head(top_n).reset_index())

    disease_counts.columns = ["Disease","Number of Trials"]


    fig = px.bar(
        disease_counts,
        x="Number of Trials",
        y="Disease",
        orientation="h",
        title="Top Disease Categories",
        text="Number of Trials"
    )

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(
        fig,
        use_container_width=True)



    st.subheader("Disease Category Distribution")

    pie_data = (
        df["source_condition_query"]
        .value_counts()
        .head(8)
        .reset_index()
    )

    pie_data.columns = [
        "Disease",
        "Count"
    ]

    fig_pie = px.pie(
        pie_data,
        names="Disease",
        values="Count",
        title="Top 8 Disease Categories"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )



    st.subheader("📌 Disease Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Unique Diseases",
            df["source_condition_query"].nunique()
        )

    with col2:
        st.metric(
            "Most Common Disease",
            df["source_condition_query"].value_counts().index[0]
        )

    with col3:
        st.metric(
            "Highest Trial Count",
            df["source_condition_query"].value_counts().iloc[0]
        )


    if "overall_status" in df.columns:

        st.subheader("Clinical Trial Status")

        status_counts = (
            df["overall_status"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        status_counts.columns = ["Status","Count"]

        fig_status = px.bar(
            status_counts,
            x="Status",
            y="Count",
            title="Clinical Trial Status Distribution"
        )

        st.plotly_chart(
            fig_status,
            use_container_width=True
        )


        st.subheader("🩺 Frequently Occurring Medical Terms")

       
        all_text = " ".join(
            df["processed_text"]
            .dropna()
            .astype(str)
        )

        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())

       
        word_counts = Counter(words)

       
        top_words = word_counts.most_common(20)

        
        word_df = pd.DataFrame(
            top_words,
            columns=["Medical Term", "Frequency"]
        )

        st.dataframe(
            word_df,
            use_container_width=True
        )

elif page == "🧠 Disease Prediction":

    st.title("🧠 Clinical Trial Disease Prediction")

    st.markdown("""
    Enter a **clinical trial summary** below.
    The trained NLP model will predict the corresponding
    disease category.
    """)


    summary = st.text_area(
        "Enter Clinical Trial Summary",
        height=250,
        placeholder="""
Example:

This clinical trial evaluates the effectiveness
and safety of a new treatment in patients with
type 2 diabetes. Participants will receive the
study medication and their blood glucose levels
will be monitored.
"""
    )

    
    if st.button("🔍 Predict Disease",type="primary"):

        if summary.strip() == "":
            st.warning("Please enter a clinical trial summary.")

        else:

            summary_tfidf = tfidf.transform([summary])

            prediction = model.predict(summary_tfidf)
            
            predicted_disease = (label_encoder.inverse_transform(prediction.astype(int)))

            disease = predicted_disease[0]

            st.success(f"Predicted Disease Category: {disease}")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("🧬 Prediction")

                st.metric(
                    "Predicted Disease",disease)

            with col2:

                st.subheader("📄 Input Summary")

                st.write(summary)


            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    summary_tfidf)[0]

                classes = label_encoder.inverse_transform(
                    np.arange(len(probabilities)))

                probability_df = pd.DataFrame({
                    "Disease": classes,
                    "Probability": probabilities
                })

                probability_df = (probability_df
                    .sort_values(
                        "Probability",
                        ascending=False
                    ).head(5))

                probability_df["Probability"] *= 100

                st.subheader(
                    "📊 Top Prediction Probabilities"
                )

                fig_prob = px.bar(
                    probability_df,
                    x="Probability",
                    y="Disease",
                    orientation="h",
                    text="Probability",
                    title="Top 5 Predicted Disease Categories"
                )

                fig_prob.update_traces(
                    texttemplate="%{text:.2f}%",
                    textposition="outside"
                )

                st.plotly_chart(
                    fig_prob,
                    use_container_width=True
                )


