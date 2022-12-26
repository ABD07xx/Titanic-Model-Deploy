import streamlit as st 
import joblib
from joblib import Parallel
import pandas as pd
from io import StringIO
import streamlit.components.v1 as stc 
import os
import numpy as np 



# wrap the string data in StringIO function
StringData = StringIO("""
        Features;Meaning;Can take Values like
        Pclass;Means Passengers class; 1 2 3
        Sex;male/female; 0 1
        Age;put your age; 23.0
        sibsp; means siblings or spouse with you it; 0 1 2 3 4 ..
        parch; means parents and children with you; 0 1 2 3 4 .. 
        Fare; any float; min = 0.0 to max 513.0
        Embarked; station ;0 ,1, 2                  
  """)
# let's read the data using the Pandas
# read_csv() function
df = pd.read_csv(StringData, sep =";")
 
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Titanic Survival Prediction</h1>
		<h4 style="color:white;text-align:center;">Logistic Regression </h4>
		</div>
		"""
stc.html(html_temp)

st.table(df)

@st.cache
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model
#loaded_model = load_model('model.pkl')

# Layout

PClass = st.slider("PClass",1,3)
Sex = st.radio("Gender",("Female","Male"))
Age = st.slider("Age",1,100)
sibsp = st.slider("Siblings+Spouse Count",0,20) 
parch = st.slider("Parents+Children Count",0,20)
fare = st.number_input("Fare(1$ to 500$ is ideal.)") 
Embarked = st.radio("Embarked",['Southampton','Cherbourg','Queenstown']) 
     
    
with st.expander("Your Selected Options"):
    result = {'Passengers Class':PClass,
    'Sex':Sex,
    'Age':Age,
    'Siblings/Spouse':sibsp,
    'Parents/Children':parch,
    'Fare':fare,
    'Embarked':Embarked
    }
    st.write(result)

gender_map   = {"Female":0,"Male":1}
embarker_map = {'Southampton':1,'Cherbourg':2,'Queenstown':3}

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 
your_data = []

for i in result.values():
    if type(i) == int or type(i)==float:
        your_data.append(i)
    elif i in ["Female","Male"]:
        res = get_value(i,gender_map)
        your_data.append(res)
    elif i in['Southampton','Cherbourg','Queenstown']:
        res1 = get_value(i,embarker_map)
        your_data.append(res1)
your_data=np.reshape(your_data,(1,-1))

if st.button("Predict"):
    st.text("Data sent to the model:")
    st.text(your_data)
    model = load_model('model.pkl')
    pred = model.predict(your_data)
    if pred == 0:
        st.error("Oh no You didn't survived")
    else:
        st.success("You survived")
    
