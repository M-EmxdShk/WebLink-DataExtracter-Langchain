import silence
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = UnstructuredURLLoader(urls=[
    "https://www.thenationalnews.com/news/mena/2026/03/01/the-supersonic-boom-of-thaads-hit-to-kill-defences-more-100km-above-uae/"
])

data=loader.load()

text= """
People underneath the cone of sound will hear a sudden crack, or boom, even if the object of the interception is very high above them.
Soundwaves can also bend back towards the ground, with differences in air temperature and density making distant explosions travel further, generating a powerful boom effect over Abu Dhabi or Dubai.
Some have described it as the “canyoning effect” experienced by troops in mountainous Afghanistan when the Taliban might open fire some distance away but the echoing effect of steep valleys made it sound much closer. 
"""

r_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "."],
    chunk_size=200
)
chunks = r_splitter.split_text(text)
#print(chunks)
#for chunk in chunks:
    #print(len(chunk))

# Setting up Vectors/Embeddings
df = pd.read_csv("sample_text.csv")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = encoder.encode(df.text) # Encodes (Basically translate)
dimensions = vectors.shape[1]
print(vectors.shape)
print(dimensions)


# FAISS & Index
index = faiss.IndexFlatL2(dimensions) # Storing Dimensions in a FAISS-made Index
index.add(vectors)

# Example
search_query="Which country to travel to"

vec = encoder.encode(search_query)
svec = np.array(vec).reshape(1,-1) # Making the array 2D
distance, I = index.search(svec, k=2) # Number of similar vectors you want to show.
print(I) # Prints Position
print(df.loc[6]) # Prints the row in the CSV file