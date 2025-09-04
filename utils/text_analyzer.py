from typing import Dict, Any
from langdetect import detect, LangDetectException
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class TextAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of the given text.
        Returns a dictionary with positive, negative, neutral, and compound scores.
        """
        return self.sia.polarity_scores(text)
    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Calculate basic text statistics.
        Returns word count, character count, and sentence count.
        """
        words = word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        
        return {
            'word_count': len(words),
            'character_count': len(text),
            'sentence_count': len(sentences),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        Returns the language code (e.g., 'en' for English).
        """
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform complete text analysis.
        Returns a dictionary with all analysis results.
        """
        if not text.strip():
            return {
                'error': 'Text cannot be empty',
                'status': 'error'
            }
            
        return {
            'status': 'success',
            'language': self.detect_language(text),
            'sentiment': self.analyze_sentiment(text),
            'statistics': self.get_text_statistics(text),
            'text': text
        }

# Create a global instance for easy access
text_analyzer = TextAnalyzer()
