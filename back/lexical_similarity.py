from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calcular_similitud_coseno(doc1_content, doc2_content):
  
    vectorizer = TfidfVectorizer()
    vectores = vectorizer.fit_transform([doc1_content, doc2_content])
    similitud_coseno = cosine_similarity(vectores[0:1], vectores[1:2])[0][0]
    return similitud_coseno

import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show(document, filename):

    wordcloud = WordCloud(width =   800, height =   800,
                    background_color ='white',
                    min_font_size =   10).generate(document)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # Guarda la imagen en una carpeta estática
    os.makedirs('static/images', exist_ok=True)
    plt.savefig(f'static/images/{filename}.png')
    plt.close()
