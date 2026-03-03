from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredURLLoader

loader = UnstructuredURLLoader(urls=[
    "https://www.thenationalnews.com/news/mena/2026/03/01/the-supersonic-boom-of-thaads-hit-to-kill-defences-more-100km-above-uae/"
])

data=loader.load()
print(data)