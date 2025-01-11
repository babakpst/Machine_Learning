
from transformers import pipeline
classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life")
print(res)


res = classifier("The weather is snowy today")
print(res)
