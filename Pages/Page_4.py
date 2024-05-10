import streamlit as st
from st_pages import add_page_title, hide_pages
from PIL import Image, ImageEnhance
from Models.HEURISTIC import heurestic
import numpy as np
from Models.init_models import *
import io
from keras.utils import load_img, img_to_array



add_page_title()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.image(uploaded_file)
col1, col2, col3, col4 = st.columns(4)

a = st.button("Submit", use_container_width=True)


with col1:
    st.header("CNN")
    if a:
        if uploaded_file is not None:
            ans = get_cnn_predict(uploaded_file)
            st.write(round(ans, 2))

with col2:
    st.header("NIMA")
    if a:
        if uploaded_file is not None:
            # img_d = uploaded_file.getvalue()
            ans = nima_pred(uploaded_file)
            st.write(round(ans, 2))


with col3:
    st.header("MN")
    if a:
        if uploaded_file is not None:
            ans = mobilenet_pred(uploaded_file)
            st.write(ans)

with col4:
    st.header("WS")
    if a:
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            original_image = np.array(original_image)
            ans = heurestic(uploaded_file)
            st.write(ans)
