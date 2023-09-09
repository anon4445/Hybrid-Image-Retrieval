import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import glob 
import torch 

torch.device('cpu')

folder_path = "images/"
image_paths = glob.glob(folder_path + "*.jpeg")
index = None
model = None
combined_embeddings = None 
labels = []

@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')

@st.cache_resource
def load_labels():
    with open(folder_path + "labels.txt", "r") as file:
        return [line.strip() for line in file.readlines()]

@st.cache_resource
def create_combined_embeddings():
    embeddings = []
    for path, label in zip(image_paths, labels):
        img = Image.open(path)
        image_embedding = model.encode(img, convert_to_tensor=True).cpu().detach().numpy()
        label_embedding = model.encode(label, convert_to_tensor=True).cpu().detach().numpy()
        combined_embedding = (image_embedding + label_embedding) / 2.0
        embeddings.append(combined_embedding)
    return np.vstack(embeddings)

@st.cache_resource
def create_faiss_index():
    faiss.normalize_L2(combined_embeddings)
    index = faiss.IndexFlatIP(combined_embeddings.shape[1])
    index.add(combined_embeddings)
    return index

model = load_model()
labels = load_labels()
combined_embeddings = create_combined_embeddings()
index = create_faiss_index()
st.write("Embedding Vector Created")
st.title("Hybrid Image Retrieval: Text or Visual Similarity Search.")
query = st.text_input("Enter search text:")
if st.button("Search"):
    if not query:
        st.warning("Please enter a search query first.")
    else:
        text_embedding = model.encode(query, convert_to_tensor=True).cpu().detach().numpy()
        text_embedding = text_embedding.reshape(1, -1)
        faiss.normalize_L2(text_embedding)
        D, indices = index.search(text_embedding, 1)  # top 1 most similar image
        matched_image_path = image_paths[indices[0][0]]
        st.write(matched_image_path)
        matched_image = Image.open(matched_image_path)
        st.image(matched_image, caption=labels[indices[0][0]], use_column_width=True) 

uploaded_image = st.file_uploader("Or, upload an image to find similar images:", type=["jpg", "png", "jpeg"])

if st.button("Search by Image") and uploaded_image:
    with Image.open(uploaded_image) as img:
        col1, col2 = st.columns(2)  
        col1.image(img, caption="Uploaded Image", use_column_width=True)
        image_embedding = model.encode(img, convert_to_tensor=True).cpu().detach().numpy()
        image_embedding = image_embedding .reshape(1, -1)
        faiss.normalize_L2(image_embedding)
        image_embedding = image_embedding.reshape(1, -1)
        D, indices = index.search(image_embedding, 1)  # top 1 most similar image
        matched_image_path = image_paths[indices[0][0]]
        matched_image = Image.open(matched_image_path)
        col2.image(matched_image, caption=labels[indices[0][0]], use_column_width=True)
if __name__ == "__main__":
    pass

