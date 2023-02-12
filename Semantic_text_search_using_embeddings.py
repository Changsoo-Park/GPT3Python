import openai
from pprint import pprint
import numpy as np
import pandas as pd


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('openaiapikey.txt')

datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

from openai.embeddings_utils import get_embedding, cosine_similarity

# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


results = search_reviews(df, "delicious beans", n=3)
print('delicious beans\n')
pprint(results)

results = search_reviews(df, "whole wheat pasta", n=3)
print('whole wheat pasta\n')
pprint(results)

results = search_reviews(df, "bad delivery", n=1)
print('bad delivery\n')
pprint(results)

results = search_reviews(df, "spoilt", n=1)
print('spoilt\n')
pprint(results)

results = search_reviews(df, "pet food", n=2)
print('pet food\n')
pprint(results)

results = search_reviews(df, "Korean food", n=2)
print('Korean food\n')
pprint(results)

results = search_reviews(df, "Kimchi", n=2)
print('Kimchi\n')
pprint(results)
