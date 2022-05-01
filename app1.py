import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
#st.sidebar.write("Setting")


st.title("brain tumer classification",)
st.header("insert ur mri image",)
[deprecation]
showfileUploaderEncoding = False
file_upload=st.file_uploader("choose the mri file",type=['jpg','png','jpeg'])

image = Image.open(file_upload)
size=(180,180)
image=ImageOps.fit(image,size,Image.ANTIALIAS)
img=np.asarray(image)
img_reshape=img[np.newaxis,...]
if file_upload is None:
    st.wtite('no file has inserted')
else: 
    st.image(img_reshape, caption='your mri image')
