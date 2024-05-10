import io
import torch
import streamlit as st
from PIL import Image, ImageEnhance
from Models.HEURISTIC import heurestic
from Models.init_models import *
import numpy as np
from st_pages import add_page_title
from PIL import Image, ImageEnhance


add_page_title()

choose = st.radio(
    "***Choose image:***",
    ["Upload", "Take photo"])
    #captions = ["?", "??", "???"])

if choose == "Upload":
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.image(uploaded_file)
        output = get_cnn_predict(uploaded_file)




elif choose == "Take photo":
    uploaded_fil = st.camera_input("Take a picture")
    ans = heurestic(uploaded_fil)
    st.write(ans)

