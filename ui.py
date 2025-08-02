import streamlit as st
import joblib

ml=joblib.load("cmnts.pkl")

sentiment={1:"Positive Feedback😊",0:"Neutral Feedback😐",-1:"Negative Feedback😒"}

vec = joblib.load("vector.pkl")

st.title("📝 Comments Analyzer")
st.slider("Rate us",1,10,7)

cmnt=st.text_area("Enter your comments ")

if(st.button("Analyze Sentiment")):
    v=vec.transform([cmnt])
    prd=ml.predict(v)[0]
    fb=sentiment.get(prd)
    st.write(fb)
    
    


