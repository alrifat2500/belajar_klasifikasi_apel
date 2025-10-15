import streamlit as st
import pandas as pd
import joblib

st.title("Klasifikasi apel")
st.markdown("model klasiifikasi apel untuk memprediksi kualitas apel")

model=joblib.load("model_klasifikasi_apel.joblib")

diameter = st.slider("Diameter",4.0, 7.0, 5.0)
berat = st.slider("Berat",150.0, 250.0, 200.0)
tebal_kulit = st.slider("Tebal kulit",0.5, 1.5, 1.0)
kadar_gula = st.slider("Kadar gula",4.0, 15.0, 10.0)
asal_daerah = st.pills("asal daerah", ["Malang", "Garut", "Boyolali"], default="Malang")
warna = st.pills("Warna", ["kuning kemerahan", "merah", "hijau"], default="kuning kemerahan")
musim_panen = st.pills("Musim panen", ["kemarau", "hujan"], default="kemarau")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,warna,musim_panen]], columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat oleh Alrifat")



	