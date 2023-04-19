import pandas as pd
import numpy as np
import sys
import gensim
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import language_check

pd.set_option("display.max_colwidth", 10000)
data = pd.read_csv('data.csv')
np.set_printoptions(threshold=sys.maxsize)

texts = data['Text']
titles = data['Title']
trainingDataForDoc2Vec = []

sid = SentimentIntensityAnalyzer()
tool = language_check.LanguageTool('en-US')

trainingDataForDoc2Vec = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
trainingDataForDoc2Vec.extend([gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(titles)])

model = gensim.models.doc2vec.Doc2Vec(trainingDataForDoc2Vec, vector_size=2000, min_count=3, epochs=40, workers=4)

model.save('Doc2VecModel')

politeScores = []
numErrors = []
postLengths = []

for text in texts:
    politeScores.append(sid.polarity_scores(text))
    numErrors.append(len(tool.check(text)))
    postLengths.append(len(text.split()))

texts = [np.array(model.infer_vector(text.split())) for text in texts]
titles = [np.array(model.infer_vector(title.split())) for title in titles]

data['Title'] = titles
data['Text'] = texts
data['Polite Score'] = politeScores
data['Num Errors'] = numErrors
data['Post Length'] = postLengths

data.to_csv('processed_data.csv')