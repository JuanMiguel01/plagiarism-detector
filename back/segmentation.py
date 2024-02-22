import spacy
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from .lexical_similarity import calcular_similitud_coseno

class SemanticVectorizer:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def vectorize(self, text):
        # Tokenizar y vectorizar el texto usando BERT
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        return last_hidden_states[0].mean(dim=0).numpy()

class PlagiarismDetector:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.tokenizer = SemanticVectorizer ()
        self.nlp = spacy.load('en_core_web_sm')

    def segment_text(self, text):
        doc = self.nlp (text)
        paragraphs = [(sent.text, sent.start, sent.end) for sent in doc.sents]
        return paragraphs

    def detect_plagiarism(self, doc1, doc2, threshold=0.85):
        segments_doc1 = self.segment_text(doc1)
        segments_doc2 = self.segment_text(doc2)

        preprocessed_segments_doc1 = [self.preprocessor.preprocess(segment[0]) for segment in segments_doc1]
        preprocessed_segments_doc2 = [self.preprocessor.preprocess(segment[0]) for segment in segments_doc2]

        vectorized_doc1 = [self.tokenizer.vectorize(segment) for segment in preprocessed_segments_doc1]
        vectorized_doc2 = [self.tokenizer.vectorize(segment) for segment in preprocessed_segments_doc2]

        plagiarism_segments = []
        for vector1, segment1 in zip(vectorized_doc1, segments_doc1):
            for vector2, segment2 in zip(vectorized_doc2, segments_doc2):

                similarity1 = cosine_similarity([vector1], [vector2])
                similarity2 = calcular_similitud_coseno(segment1[0], segment2[0])
                similarity = (similarity1 + similarity2) / 2

                if similarity[0][0] > threshold:
                    original_segment1 = self.nlp(doc1)[segment1[1]:segment1[2]]
                    original_segment2 = self.nlp(doc2)[segment2[1]:segment2[2]]
                    plagiarism_segments.append((original_segment1, original_segment2, similarity[0][0]))
        
     
        plagiarism_segments = sorted(plagiarism_segments, key=lambda x: x[2], reverse=True)

        return plagiarism_segments

