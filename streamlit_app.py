from pathlib import Path

import streamlit as st
from PIL import Image
from st_pages.__init__ import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page

add_page_title()
# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("streamlit_app.py", "About...", "🏠"),
        Page("Pages/Page_2.py", "Image", ":rainbow:"),
        Page("Pages/Page_3.py", "Methods", ":sparkles:"),
        Page("Pages/Page_4.py", "Compare methods", ":part_alternation_mark:"),
    ]
)

st.header("Методы вычисления эстетической привлекательности изображений")
st.write("***Image Quality Assessment (IQA).***")
st.write("Цель задачи: Представить к сравнению различные модели, способные в автоматическом режиме наиболее "
         "корректно предсказывать оценку эстетической привлекательности изображения. "
         "Иными словами мы полагаем, что смазанным, нечетким, шумным "
         "изображениям в результате работы модели будет проставляться более низкая оценка, "
         "в то время как изображениям красочным, с высоким качеством и детализацией "
         "более высокая.")



def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url("/vk_logo.png");
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "VK-HSE project";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
add_logo()