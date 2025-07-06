import streamlit as st
import pandas as pd
import joblib
import altair as alt
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Heart Disease Diagnosis", layout="wide")
st.title("Heart Disease Prediction - Voting Classifier")

model = joblib.load("best_model.pkl")

features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

def preprocess_input(df):
    df = df.copy()
    df = df[[col for col in df.columns if col in features]]
    mappings = {
        'Sex': {'M': 1, 'F': 0},
        'ChestPainType': {'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3},
        'RestingECG': {'Normal': 0, 'LVH': 1, 'ST': 2},
        'ExerciseAngina': {'N': 0, 'Y': 1},
        'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

mode = st.radio("Choose input method:", ["Manual input", "Upload CSV file"])

if mode == "Manual input":
    st.subheader("Enter patient details:")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 0, 100, 45)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"])
        resting_bp = st.number_input("Resting Blood Pressure", value=120)
        cholesterol = st.number_input("Cholesterol", value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar (greater than 120 mg/dl)", [0, 1])

    with col2:
        rest_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "LVH", "ST"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", value=1.0)
        st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Up", "Flat", "Down"])

    input_dict = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': rest_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([input_dict])
    processed_df = preprocess_input(input_df)

    if st.button("Diagnose the case"):
        prediction = model.predict(processed_df)[0]
        label = "No Heart Disease" if prediction == 0 else "Possible Heart Disease Detected"
        st.subheader("Diagnosis Result:")
        st.success(label)
        st.dataframe(input_df)

else:
    st.subheader("Upload CSV file for bulk diagnosis")
    uploaded_file = st.file_uploader("Upload file here", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        original_data = data.copy()
        process_data = data[[col for col in data.columns if col in features]]

        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

        try:
            processed = preprocess_input(process_data)
            imputer = SimpleImputer(strategy='mean')
            processed[numerical_cols] = imputer.fit_transform(processed[numerical_cols])
            predictions = model.predict(processed)

            original_data = original_data.loc[processed.index]
            original_data.insert(0, "person_id", range(1, len(original_data) + 1))
            original_data["heart_disease_prediction"] = predictions
            original_data["Patient Diagnosis"] = original_data["heart_disease_prediction"].replace({0: "No Disease", 1: "Possible Disease"})

            st.success("Prediction completed successfully")

            st.subheader("Complete table of all cases")
            st.dataframe(original_data)
            total_count = len(original_data)
            count_disease = (original_data["heart_disease_prediction"] == 1).sum()
            count_healthy = (original_data["heart_disease_prediction"] == 0).sum()
            st.caption(f"🧮 Total records: {total_count} | With disease: {count_disease} ({(count_disease/total_count)*100:.2f}%) | Without disease: {count_healthy} ({(count_healthy/total_count)*100:.2f}%)")

            st.subheader("Patients predicted with possible heart disease")
            df_disease = original_data[original_data["heart_disease_prediction"] == 1]
            st.dataframe(df_disease)
            st.caption(f"📊 Patients with disease: {len(df_disease)} / {total_count} ({(len(df_disease)/total_count)*100:.2f}%)")

            st.subheader("Patients predicted as healthy")
            df_healthy = original_data[original_data["heart_disease_prediction"] == 0]
            st.dataframe(df_healthy)
            st.caption(f"🧘 Patients without disease: {len(df_healthy)} / {total_count} ({(len(df_healthy)/total_count)*100:.2f}%)")

            chart_type = st.selectbox("Select chart type:", ["Bar Chart", "Area Chart", "Line Chart", "Circle Pack Chart"])

            def plot_chart(data, col, chart_type):
                if chart_type == "Bar Chart":
                    chart = alt.Chart(data).mark_bar().encode(
                        x=alt.X(f"{col}:N" if data[col].dtype == 'object' else f"{col}:Q", title=col),
                        y='count()',
                        color='Patient Diagnosis:N',
                        tooltip=['count()']
                    ).interactive().properties(width=600)
                elif chart_type == "Area Chart":
                    chart = alt.Chart(data).mark_area(
                        opacity=0.4, interpolate='step'
                    ).encode(
                        x=alt.X(f"{col}:Q" if data[col].dtype != 'object' else f"{col}:N",
                                bin=alt.Bin(maxbins=40) if data[col].dtype != 'object' else None, title=col),
                        y='count()',
                        color='Patient Diagnosis:N',
                        tooltip=['count()']
                    ).interactive().properties(width=600)
                elif chart_type == "Line Chart":
                    chart = alt.Chart(data).mark_line(point=True).encode(
                        x=alt.X(f"{col}:Q" if data[col].dtype != 'object' else f"{col}:N",
                                bin=alt.Bin(maxbins=40) if data[col].dtype != 'object' else None, title=col),
                        y='count()',
                        color='Patient Diagnosis:N',
                        tooltip=['count()']
                    ).interactive().properties(width=600)
                elif chart_type == "Circle Pack Chart":
                    chart = alt.Chart(data).mark_circle().encode(
                        x=alt.X(f"{col}:N" if data[col].dtype == 'object' else f"{col}:Q", title=col),
                        y='count()',
                        size='count()',
                        color='Patient Diagnosis:N',
                        tooltip=['count()']
                    ).interactive().properties(width=600)
                else:
                    chart = None
                return chart

            for col in categorical_cols + numerical_cols:
                if col in original_data.columns:
                    st.markdown(f"### Distribution of `{col}` based on diagnosis result")
                    chart = plot_chart(original_data, col, chart_type)
                    if chart:
                        st.altair_chart(chart)

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("""
---
<div style='text-align:center; color:gray; font-size:13px;'>
    Developed by <b>Mohamed Dyaa</b><br>
</div>
""", unsafe_allow_html=True)


# streamlit run "D:\mohamed dyaa\2.heart_disease\app.py"