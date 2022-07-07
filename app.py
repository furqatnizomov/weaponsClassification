from  fastai.vision.all import *
import streamlit as st
import pathlib

# OSga moslash
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath=pathlib.PosixPath

#title
st.title("Menkim...")

#description
st.text("""Assalomu alaykum, qurollarni klassifikatsiya qiluvchi modelman, hozircha faqat
yuklagan rasmingizni taniy olaman, katta bo'lsam, videodan ham tanib olaman
degan niyatdaman.
Quyidagi qurollarning rasmini yuklashingiz mumkin:
* Bomb 
* Missile 
* Sword 
* Handgun 
* Tank 
* Axe 
* Shotgun 
* Knife 
* Dagger 
* Bow and arrow 
* Cannon""",)

#rasmni joylash
file = st.file_uploader("Rasm yuklash",type=['jpg','jpeg','gif','svg'])
if file:
    st.markdown('Natija bilan tanishamiz:')
    st.text("bu siz yuklagan rasm")

    #PIL convert
    img = PILImage.create(file)
    st.image(file)

    #model
    model = load_learner("weapon_model_200.pkl")

    #bashorat
    pred, pred_id, probs = model.predict(img)
    st.text("Rasmdagi qurolning nomi: ")
    st.success(pred)
    st.text("Ehtimolligi: ")
    st.info(f"{probs[pred_id]*100:.1f}%")
    