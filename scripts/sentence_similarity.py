# scripts/sentence_similarity.py

import nltk
from sklearn.metrics import jaccard_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Ensure NLTK data is downloaded
nltk.download('punkt')

def jaccard_similarity(sent1, sent2):
    """
    Calculates the Jaccard similarity between two sentences.
    
    Parameters:
    - sent1 (str): The first sentence.
    - sent2 (str): The second sentence.
    
    Returns:
    - float: Jaccard similarity score.
    """
    # Tokenize the sentences
    tokens1 = set(nltk.word_tokenize(sent1.lower()))
    tokens2 = set(nltk.word_tokenize(sent2.lower()))
    
    # Calculate intersection and union
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    # Handle division by zero
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def bleu_score_func(reference, candidate):
    """
    Calculates the BLEU score between a reference and a candidate sentence using smoothing.
    
    Parameters:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.
    
    Returns:
    - float: BLEU score.
    """
    # Tokenize the sentences
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # Handle empty candidate
    if not candidate_tokens:
        return 0.0
    
    # Initialize smoothing function
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU score with smoothing
    return sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5), smoothing_function=smoothie)

def rouge_score_func(reference, candidate):
    """
    Calculates the ROUGE scores between a reference and a candidate sentence.
    
    Parameters:
    - reference (str): The reference sentence.
    - candidate (str): The candidate sentence.
    
    Returns:
    - dict: ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        'ROUGE1': scores['rouge1'].fmeasure,
        'ROUGE2': scores['rouge2'].fmeasure,
        'ROUGEL': scores['rougeL'].fmeasure
    }

class SimilarityCalculator:
    """
    A class to calculate various sentence similarity metrics with optimized embedding computations.
    """
    def __init__(self):
        # Initialize the SentenceTransformer model for cosine similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient and optimized
        self.embeddings_cache = {}
    
    def get_embedding(self, sentence):
        """
        Retrieves the embedding for a sentence, utilizing caching to avoid redundant computations.
        
        Parameters:
        - sentence (str): The sentence to encode.
        
        Returns:
        - numpy.ndarray: The sentence embedding.
        """
        if sentence in self.embeddings_cache:
            return self.embeddings_cache[sentence]
        else:
            embedding = self.model.encode(sentence, convert_to_numpy=True)
            self.embeddings_cache[sentence] = embedding
            return embedding
    
    def cosine_similarity(self, sentence1, sentence2):
        """
        Calculates cosine similarity between two sentences using cached embeddings.
        
        Parameters:
        - sentence1 (str): The first sentence.
        - sentence2 (str): The second sentence.
        
        Returns:
        - float: Cosine similarity score.
        """
        vec1 = self.get_embedding(sentence1)
        vec2 = self.get_embedding(sentence2)
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cos_sim
    
    def calculate_all(self, reference, candidate):
        """
        Calculates all similarity metrics between two sentences.
        
        Parameters:
        - reference (str): The reference sentence.
        - candidate (str): The candidate sentence.
        
        Returns:
        - dict: All similarity scores.
        """
        jaccard = jaccard_similarity(reference, candidate)
        bleu = bleu_score_func(reference, candidate)
        rouge = rouge_score_func(reference, candidate)
        cosine = self.cosine_similarity(reference, candidate)
        
        return {
            'Jaccard': jaccard,
            'BLEU': bleu,
            'ROUGE1': rouge['ROUGE1'],
            'ROUGE2': rouge['ROUGE2'],
            'ROUGEL': rouge['ROUGEL'],
            'Cosine_Similarity': cosine
        }

