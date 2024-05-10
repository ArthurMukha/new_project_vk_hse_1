import streamlit as st
from st_pages import add_page_title
from Models.HEURISTIC import heurestic
add_page_title()

choose = st.radio(
    "***Method:***",
    ["CNN", "Brisque", "NIMA", "CNN_MN", "Weighted score"],
    captions = ["", "", "", "MobileNet", ""])

if choose == "CNN":
    st.write("Information about method...")
    st.write("CNN")
elif choose == "NIMA":
    st.write("NIMA")
elif choose == "Brisque":
    st.write("Brisque")
elif choose == "CNN_MN":
    st.write("CNN_MN")
elif choose == "Weighted score":
    st.write("Weighted score")
    # if uploaded_file is not None:
    #     ans = heurestic(uploaded_file)
    #     st.write(ans)



# else:
#     st.write("You didn't select comedy.")

