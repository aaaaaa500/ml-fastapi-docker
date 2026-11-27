import joblib
import streamlit as st

model = joblib.load("model.joblib")
class_names = ["setosa", "versicolor", "virginica"]

st.title("Iris Flower Classifier")
st.write("Enter the flower sizes in cm and press Predict")

sepal_length = st.number_input("Sepal length", min_value=0.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal width", min_value=0.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal length", min_value=0.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal width", min_value=0.0, value=0.2, step=0.1)

if st.button("Predict"):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(data)
    class_id = int(prediction[0])
    st.success(f"Predicted class: {class_names[class_id]}")