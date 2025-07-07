import streamlit as st
import pandas as pd
import joblib
import altair as alt
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Heart Disease Diagnosis", layout="wide")
st.title("Heart Disease Prediction - Voting Classifier")

try:
    heart_disease_model = joblib.load("be_model.pkl")
except Exception as error:
    st.error(f"❌ Model loading error: {error}")
    st.stop()

input_features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

def preprocess_patient_data(patient_data):
    patient_data = patient_data.copy()
    patient_data = patient_data[[column for column in patient_data.columns if column in input_features]]

    feature_mappings = {
        'Sex': {'Male': 1, 'Female': 0},
        'ChestPainType': {
            'Asymptomatic': 0,
            'Non-anginal Pain': 1,
            'Atypical Angina': 2,
            'Typical Angina': 3
        },
        'RestingECG': {'Normal': 0, 'LVH': 1, 'ST': 2},
        'ExerciseAngina': {'No': 0, 'Yes': 1},
        'ST_Slope': {'Upward': 0, 'Flat': 1, 'Downward': 2}
    }

    for column_name, mapping in feature_mappings.items():
        if column_name in patient_data.columns:
            patient_data[column_name] = patient_data[column_name].map(mapping)
            patient_data[column_name].fillna(-1, inplace=True)

    return patient_data

input_mode = st.radio("Choose input method:", ["Manual input", "Upload CSV file"])

if input_mode == "Manual input":
    st.subheader("Enter patient details:")
    left_column, right_column = st.columns(2)

    with left_column:
        patient_age = st.slider("Age", 0, 100, 45)
        patient_sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain_type = st.selectbox("Chest Pain Type", [
            "Asymptomatic", "Non-anginal Pain", "Atypical Angina", "Typical Angina"
        ])
        resting_blood_pressure = st.number_input("Resting Blood Pressure", value=120)
        cholesterol_level = st.number_input("Cholesterol", value=200)
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar (greater than 120 mg/dl)", [0, 1])

    with right_column:
        resting_electrocardiographic_result = st.selectbox("Resting Electrocardiographic Results", ["Normal", "LVH", "ST"])
        maximum_heart_rate = st.number_input("Maximum Heart Rate Achieved", value=150)
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak_value = st.number_input("Oldpeak (ST Depression)", value=1.0)
        st_segment_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upward", "Flat", "Downward"])

    single_patient_input = {
        'Age': patient_age,
        'Sex': patient_sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_blood_pressure,
        'Cholesterol': cholesterol_level,
        'FastingBS': fasting_blood_sugar,
        'RestingECG': resting_electrocardiographic_result,
        'MaxHR': maximum_heart_rate,
        'ExerciseAngina': exercise_induced_angina,
        'Oldpeak': oldpeak_value,
        'ST_Slope': st_segment_slope
    }

    patient_input_dataframe = pd.DataFrame([single_patient_input])
    preprocessed_input = preprocess_patient_data(patient_input_dataframe)

    if st.button("Diagnose the case"):
        diagnosis_prediction = heart_disease_model.predict(preprocessed_input)[0]
        diagnosis_result = "No Heart Disease" if diagnosis_prediction == 0 else "Possible Heart Disease Detected"
        st.subheader("Diagnosis Result:")
        st.success(diagnosis_result)
        st.dataframe(patient_input_dataframe)

else:
    st.subheader("Upload CSV file for bulk diagnosis")
    uploaded_csv_file = st.file_uploader("Upload file here", type="csv")

    if uploaded_csv_file is not None:
        uploaded_data = pd.read_csv(uploaded_csv_file)
        original_uploaded_data = uploaded_data.copy()

        missing_columns = [column for column in input_features if column not in uploaded_data.columns]
        if missing_columns:
            st.error(f"Missing columns in the uploaded CSV: {missing_columns}")
        else:
            filtered_uploaded_data = uploaded_data[input_features]

            categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
            numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

            try:
                processed_uploaded_data = preprocess_patient_data(filtered_uploaded_data)
                imputer = SimpleImputer(strategy='mean')
                processed_uploaded_data[numerical_columns] = imputer.fit_transform(processed_uploaded_data[numerical_columns])

                predictions = heart_disease_model.predict(processed_uploaded_data)

                original_uploaded_data = original_uploaded_data.loc[processed_uploaded_data.index]
                original_uploaded_data.insert(0, "person_id", range(1, len(original_uploaded_data) + 1))
                original_uploaded_data["heart_disease_prediction"] = predictions
                original_uploaded_data["Patient Diagnosis"] = original_uploaded_data["heart_disease_prediction"].replace({0: "No Disease", 1: "Possible Disease"})

                st.success("Prediction completed successfully")

                st.subheader("Complete table of all cases")
                st.dataframe(original_uploaded_data)

                total_cases = len(original_uploaded_data)
                total_disease_cases = (original_uploaded_data["heart_disease_prediction"] == 1).sum()
                total_healthy_cases = (original_uploaded_data["heart_disease_prediction"] == 0).sum()

                st.caption(f"🧮 Total records: {total_cases} | With disease: {total_disease_cases} ({(total_disease_cases/total_cases)*100:.2f}%) | Without disease: {total_healthy_cases} ({(total_healthy_cases/total_cases)*100:.2f}%)")

                st.subheader("Patients predicted with possible heart disease")
                patients_with_disease = original_uploaded_data[original_uploaded_data["heart_disease_prediction"] == 1]
                st.dataframe(patients_with_disease)

                st.subheader("Patients predicted as healthy")
                healthy_patients = original_uploaded_data[original_uploaded_data["heart_disease_prediction"] == 0]
                st.dataframe(healthy_patients)

                selected_chart_type = st.selectbox("Select chart type:", ["Bar Chart", "Area Chart", "Line Chart", "Circle Pack Chart"])

                def plot_distribution_chart(dataframe, column_name, chart_type):
                    if chart_type == "Bar Chart":
                        chart = alt.Chart(dataframe).mark_bar().encode(
                            x=alt.X(f"{column_name}:N" if dataframe[column_name].dtype == 'object' else f"{column_name}:Q", title=column_name),
                            y='count()',
                            color='Patient Diagnosis:N',
                            tooltip=['count()']
                        ).interactive().properties(width=600)
                    elif chart_type == "Area Chart":
                        chart = alt.Chart(dataframe).mark_area(opacity=0.4, interpolate='step').encode(
                            x=alt.X(f"{column_name}:Q" if dataframe[column_name].dtype != 'object' else f"{column_name}:N",
                                    bin=alt.Bin(maxbins=40) if dataframe[column_name].dtype != 'object' else None, title=column_name),
                            y='count()',
                            color='Patient Diagnosis:N',
                            tooltip=['count()']
                        ).interactive().properties(width=600)
                    elif chart_type == "Line Chart":
                        chart = alt.Chart(dataframe).mark_line(point=True).encode(
                            x=alt.X(f"{column_name}:Q" if dataframe[column_name].dtype != 'object' else f"{column_name}:N",
                                    bin=alt.Bin(maxbins=40) if dataframe[column_name].dtype != 'object' else None, title=column_name),
                            y='count()',
                            color='Patient Diagnosis:N',
                            tooltip=['count()']
                        ).interactive().properties(width=600)
                    elif chart_type == "Circle Pack Chart":
                        chart = alt.Chart(dataframe).mark_circle().encode(
                            x=alt.X(f"{column_name}:N" if dataframe[column_name].dtype == 'object' else f"{column_name}:Q", title=column_name),
                            y='count()',
                            size='count()',
                            color='Patient Diagnosis:N',
                            tooltip=['count()']
                        ).interactive().properties(width=600)
                    else:
                        chart = None
                    return chart

                for column in categorical_columns + numerical_columns:
                    if column in original_uploaded_data.columns:
                        st.markdown(f"### Distribution of `{column}` based on diagnosis result")
                        chart_output = plot_distribution_chart(original_uploaded_data, column, selected_chart_type)
                        if chart_output:
                            st.altair_chart(chart_output)

            except Exception as error:
                st.error(f"Prediction error: {error}")

st.markdown("""
---
<div style='text-align:center; color:gray; font-size:13px;'>
    Developed by <b>Mohamed Dyaa</b><br>
</div>
""", unsafe_allow_html=True)
