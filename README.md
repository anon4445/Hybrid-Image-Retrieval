# Hybrid Image Retrieval: Text or Visual Similarity Search 

## Overview
*This project uses a hybrid approach to image retrieval. It uses CLIP (Contrastive Language–Image Pre-training) embeddings to represent both images and their associated textual labels. By averaging the embeddings of images and labels, it creates a "hybrid" embedding. This makes it possible to retrieve images by searching for either text or another image, leveraging the strengths of both visual and textual representations.*

## Dependencies
- streamlit: To create a user-friendly web interface.
- numpy: For numerical operations on arrays.
- faiss: Efficient similarity search and clustering of dense vectors.
- sentence_transformers: Provides the SentenceTransformer class for generating embeddings.
- PIL: For opening and manipulating image files.
- glob: To retrieve filenames matching a specified pattern.
- torch: For specifying the device for model operations. 

## Demo 
![Demo of the image retrival.](/assets/images/demo1.jpg)

## Working Principle

### 1. CLIP (Contrastive Language–Image Pre-training):
CLIP is a neural network model trained by OpenAI to understand images paired with natural language. In essence, it's taught to relate textual descriptions with their associated visual counterparts. It can generate embeddings for both text and images such that similar items (across both domains) end up closer in the embedding space. This means semantically similar images and text phrases will have embeddings that are near each other.
*In this project, CLIP is accessed through SentenceTransformer('clip-ViT-B-32'), which wraps the functionality and provides a method to easily get embeddings.*

### 2. Combined Embeddings:
Given the capability of CLIP to embed both text and images, the project calculates embeddings for each image and its label. The hybrid representation is created by averaging the two embeddings. This ensures that the hybrid representation captures both the visual content of the image and its textual description.
### 3. FAISS (Facebook AI Similarity Search):
FAISS is a library built by Facebook AI that allows for efficient similarity searches of vectors. Before using FAISS, the vectors (in this case, the combined embeddings) are normalized to have a unit length. This normalization facilitates the cosine similarity measure to be computed using inner product. The normalized combined embeddings are then indexed in FAISS. 
*This project utilizes IndexFlatIP which is designed for inner product similarity (equivalent to cosine similarity for normalized vectors).*

### 4. Streamlit Interface:
Streamlit is a Python library that lets you create web apps for machine learning and data science.
*In this project, Streamlit provides an interactive front end. Users can input a text or upload an image to search for similar items in the dataset.* 

### 5. Query Processing:
When a user inputs a text or image:
- The text or image is passed through the CLIP model to get its embedding.
- This embedding is then normalized and searched against the FAISS index.
- The FAISS index returns the most similar item (or items) from the dataset.
- Finally, the corresponding image (based on similarity) is displayed in the Streamlit interface.

### Principle Summary:
At its core, the system translates both visual and textual data into a shared embedding space using the CLIP model. These embeddings are indexed using FAISS for efficient similarity searches. When a user queries with either text or an image, the system finds the most similar item from the dataset in this shared embedding space and presents the result. The use of combined embeddings allows for meaningful similarity searches across both visual and textual domains.

## How to Run 
Clone the repo 
```
git clone https://github.com/jakariaemon/Text-to-Image-Search.git 
```
Install requirements:  
```
pip install -r requirements.txt
``` 
Run the server and a browser window will open: 
```
streamlit run server.py 
``` 

