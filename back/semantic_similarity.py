from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from .preproccesing import TextPreprocessor

# Crear una instancia de TextPreprocessor
preprocessor = TextPreprocessor(language='english')

def train_word2vec(texts, vector_size=100, window_size=5, min_count=1, epochs=10):
    processed_texts = [preprocessor.tokenize_text(text) for text in texts]
    model = Word2Vec(processed_texts, vector_size=vector_size, window=window_size, min_count=min_count, epochs=epochs)
    return model

def get_document_vector(model, document):
    processed_document = preprocessor.tokenize_text(document)
    document_vector = sum([model.wv[word] for word in processed_document])
    return document_vector / len(processed_document)    

def semantic_similarity(model, document1, document2):
    vector1 = get_document_vector(model, document1)
    vector2 = get_document_vector(model, document2)
    return cosine_similarity([vector1], [vector2])[0][0]


def calcular_similitud_semantica(doc1_content, doc2_content):

    docs = [doc1_content, doc2_content]
    model = train_word2vec(docs)

    similarity = semantic_similarity(model, doc1_content, doc2_content)

    return similarity
