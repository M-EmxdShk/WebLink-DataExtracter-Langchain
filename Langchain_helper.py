import silence
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

