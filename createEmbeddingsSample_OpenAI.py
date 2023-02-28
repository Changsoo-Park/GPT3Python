import openai
from pprint import pprint
import numpy as np

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('openaiapikey.txt')

response = openai.Embedding.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter..."
)

embeddings = response['data'][0]['embedding']

pprint(embeddings)


