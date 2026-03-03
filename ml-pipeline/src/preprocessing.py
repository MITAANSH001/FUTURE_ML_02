"""
Text Preprocessing Module

Handles text cleaning, tokenization, and feature extraction for support tickets.
Implements industry-standard NLP preprocessing techniques.
"""

import re
import string
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder


class TextPreprocessor:
    """Preprocess and vectorize text data for ML models."""

    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
        'the', 'to', 'was', 'will', 'with', 'i', 'me', 'my', 'you', 'your',
        'we', 'us', 'they', 'them', 'what', 'which', 'who', 'when', 'where',
        'why', 'how', 'can', 'could', 'would', 'should', 'may', 'might'
    }

    def __init__(self):
        """Initialize preprocessor."""
        self.tfidf_vectorizer = None
        self.label_encoders = {}

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and digits (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return text.split()

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        """
        Remove common stopwords.

        Args:
            tokens: List of tokens

        Returns:
            Filtered tokens
        """
        return [token for token in tokens if token not in TextPreprocessor.STOPWORDS and len(token) > 2]

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Complete preprocessing pipeline.

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        # Clean text
        text = TextPreprocessor.clean_text(text)

        # Tokenize
        tokens = TextPreprocessor.tokenize(text)

        # Remove stopwords
        tokens = TextPreprocessor.remove_stopwords(tokens)

        # Rejoin tokens
        return ' '.join(tokens)

    def fit_tfidf(self, texts: List[str], max_features: int = 500) -> None:
        """
        Fit TF-IDF vectorizer on texts.

        Args:
            texts: List of text documents
            max_features: Maximum number of features to extract
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_vectorizer.fit(texts)

    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted TF-IDF vectorizer.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF feature matrix
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        return self.tfidf_vectorizer.transform(texts).toarray()

    def fit_transform_tfidf(self, texts: List[str], max_features: int = 500) -> np.ndarray:
        """
        Fit and transform texts using TF-IDF.

        Args:
            texts: List of text documents
            max_features: Maximum number of features

        Returns:
            TF-IDF feature matrix
        """
        self.fit_tfidf(texts, max_features)
        return self.transform_tfidf(texts)

    def encode_labels(self, labels: List[str], label_type: str = "category") -> Tuple[np.ndarray, Dict]:
        """
        Encode categorical labels to integers.

        Args:
            labels: List of labels
            label_type: Type of label (for tracking encoder)

        Returns:
            Tuple of (encoded labels, mapping dict)
        """
        if label_type not in self.label_encoders:
            self.label_encoders[label_type] = LabelEncoder()
            self.label_encoders[label_type].fit(labels)

        encoded = self.label_encoders[label_type].transform(labels)

        # Create mapping
        mapping = {
            i: label for i, label in enumerate(self.label_encoders[label_type].classes_)
        }

        return encoded, mapping

    def decode_labels(self, encoded: np.ndarray, label_type: str = "category") -> List[str]:
        """
        Decode integer labels back to original labels.

        Args:
            encoded: Encoded labels
            label_type: Type of label

        Returns:
            List of original labels
        """
        if label_type not in self.label_encoders:
            raise ValueError(f"No encoder found for label type: {label_type}")

        return self.label_encoders[label_type].inverse_transform(encoded)

    @staticmethod
    def get_feature_importance(vectorizer: TfidfVectorizer, top_n: int = 20) -> Dict[str, float]:
        """
        Get top features from TF-IDF vectorizer.

        Args:
            vectorizer: Fitted TF-IDF vectorizer
            top_n: Number of top features to return

        Returns:
            Dictionary of feature names and their importance scores
        """
        feature_names = vectorizer.get_feature_names_out()
        # Get mean importance across all documents
        feature_importance = np.asarray(vectorizer.idf_).ravel()

        # Get top features
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        top_features = {
            feature_names[i]: float(feature_importance[i])
            for i in top_indices
        }

        return top_features

    @staticmethod
    def calculate_text_statistics(texts: List[str]) -> Dict:
        """
        Calculate statistics about text data.

        Args:
            texts: List of text documents

        Returns:
            Dictionary of statistics
        """
        lengths = [len(text.split()) for text in texts]
        all_words = []
        for text in texts:
            all_words.extend(text.split())

        word_freq = Counter(all_words)

        return {
            "total_documents": len(texts),
            "total_words": len(all_words),
            "unique_words": len(word_freq),
            "avg_words_per_doc": np.mean(lengths),
            "min_words": min(lengths),
            "max_words": max(lengths),
            "median_words": np.median(lengths),
            "most_common_words": dict(word_freq.most_common(10))
        }


class FeatureExtractor:
    """Extract additional features from support tickets."""

    @staticmethod
    def extract_features(ticket_text: str) -> Dict[str, float]:
        """
        Extract hand-crafted features from ticket text.

        Args:
            ticket_text: Support ticket text

        Returns:
            Dictionary of features
        """
        text_lower = ticket_text.lower()

        return {
            "text_length": len(ticket_text),
            "word_count": len(ticket_text.split()),
            "avg_word_length": np.mean([len(word) for word in ticket_text.split()]) if ticket_text.split() else 0,
            "punctuation_count": sum(1 for char in ticket_text if char in string.punctuation),
            "uppercase_ratio": sum(1 for char in ticket_text if char.isupper()) / len(ticket_text) if ticket_text else 0,
            "question_mark_count": text_lower.count('?'),
            "exclamation_count": text_lower.count('!'),
            "digit_count": sum(1 for char in ticket_text if char.isdigit()),
        }

    @staticmethod
    def extract_batch_features(texts: List[str]) -> np.ndarray:
        """
        Extract features for multiple texts.

        Args:
            texts: List of text documents

        Returns:
            Feature matrix
        """
        features = []
        for text in texts:
            feature_dict = FeatureExtractor.extract_features(text)
            features.append(list(feature_dict.values()))

        return np.array(features)


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "I was charged twice for my subscription this month. Can you help?",
        "The application keeps crashing when I try to upload files.",
        "I forgot my password and can't reset it.",
        "How do I use the advanced filtering options?"
    ]

    preprocessor = TextPreprocessor()

    # Preprocess texts
    print("Original texts:")
    for text in sample_texts:
        print(f"  - {text}")

    print("\nPreprocessed texts:")
    preprocessed = [preprocessor.preprocess_text(text) for text in sample_texts]
    for text in preprocessed:
        print(f"  - {text}")

    # TF-IDF transformation
    print("\nTF-IDF Vectorization:")
    tfidf_matrix = preprocessor.fit_transform_tfidf(preprocessed, max_features=20)
    print(f"  Shape: {tfidf_matrix.shape}")
    print(f"  Top features: {list(preprocessor.tfidf_vectorizer.get_feature_names_out()[:10])}")

    # Text statistics
    print("\nText Statistics:")
    stats = TextPreprocessor.calculate_text_statistics(preprocessed)
    for key, value in stats.items():
        print(f"  {key}: {value}")
