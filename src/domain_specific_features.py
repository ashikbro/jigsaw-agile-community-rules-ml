import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any, Set, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
import json
from pathlib import Path

class CommunityModerationFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature extractor for community moderation.
    Focuses on detecting patterns specific to rule violations in online communities.
    """
    
    def __init__(self):
        """
        Initialize the community moderation feature extractor.
        """
        # Comprehensive banned words and phrases
        self.banned_words = {
            'profanity': [
                'damn', 'hell', 'crap', 'stupid', 'idiot', 'moron', 'dumb', 'loser',
                'pathetic', 'worthless', 'useless', 'garbage', 'trash', 'scum'
            ],
            'hate_speech': [
                'hate', 'racist', 'racism', 'sexist', 'sexism', 'homophobic', 'transphobic',
                'bigot', 'nazi', 'fascist', 'supremacist', 'discrimination'
            ],
            'threats': [
                'kill', 'die', 'death', 'murder', 'hurt', 'harm', 'violence', 'attack',
                'destroy', 'eliminate', 'annihilate', 'beat up', 'punch', 'shoot'
            ],
            'harassment': [
                'harass', 'harassment', 'bully', 'bullying', 'stalk', 'stalking',
                'intimidate', 'threaten', 'abuse', 'torment', 'persecute'
            ],
            'doxxing': [
                'address', 'phone number', 'real name', 'workplace', 'school',
                'personal info', 'doxx', 'dox', 'expose', 'reveal identity'
            ],
            'spam_indicators': [
                'buy now', 'click here', 'free money', 'get rich', 'make money fast',
                'limited time', 'act now', 'special offer', 'discount', 'sale'
            ]
        }
        
        # Moderation-specific terms
        self.moderation_terms = {
            'authority': [
                'mod', 'mods', 'moderator', 'moderators', 'admin', 'admins',
                'administrator', 'staff', 'owner', 'creator'
            ],
            'rules_policy': [
                'rule', 'rules', 'guideline', 'guidelines', 'policy', 'policies',
                'terms', 'conditions', 'regulation', 'law', 'legal'
            ],
            'violations': [
                'ban', 'banned', 'kick', 'kicked', 'remove', 'removed', 'delete',
                'deleted', 'suspend', 'suspended', 'mute', 'muted', 'timeout'
            ],
            'reporting': [
                'report', 'reported', 'flag', 'flagged', 'complaint', 'violation',
                'inappropriate', 'offensive', 'against rules'
            ]
        }
        
        # Threat escalation patterns
        self.threat_patterns = [
            r'\bi will\s+(?:kill|hurt|harm|destroy)\b',
            r'\byou should\s+(?:die|kill yourself)\b',
            r'\bi hope you\s+(?:die|get hurt)\b',
            r'\bi\'m going to\s+(?:kill|hurt|harm)\b',
            r'\byou deserve to\s+(?:die|suffer)\b',
            r'\bwatch your back\b',
            r'\byou\'re dead\b',
            r'\bi know where you live\b'
        ]
        
        # Harassment patterns
        self.harassment_patterns = [
            r'\bshut up\b',
            r'\bno one cares\b',
            r'\bkill yourself\b',
            r'\bgo away\b',
            r'\byou\'re annoying\b',
            r'\bstop posting\b',
            r'\bleave this place\b'
        ]
        
        # Politeness indicators (positive and negative)
        self.politeness_indicators = {
            'positive': [
                'please', 'thank you', 'thanks', 'sorry', 'excuse me', 'pardon',
                'appreciate', 'grateful', 'respect', 'understand', 'agree',
                'good point', 'well said', 'interesting', 'helpful'
            ],
            'negative': [
                'shut up', 'whatever', 'don\'t care', 'boring', 'waste of time',
                'ridiculous', 'nonsense', 'absurd', 'pathetic', 'lame'
            ]
        }
        
        # Community-specific violation patterns
        self.community_patterns = {
            'off_topic': [
                r'\boff.?topic\b',
                r'\bnot related\b',
                r'\bwrong sub\b',
                r'\bdoesn\'t belong\b'
            ],
            'self_promotion': [
                r'\bcheck out my\b',
                r'\bvisit my\b',
                r'\bsubscribe to\b',
                r'\bfollow me\b',
                r'\bmy channel\b',
                r'\bmy website\b'
            ],
            'trolling': [
                r'\btroll\b',
                r'\btrolling\b',
                r'\bbaiting\b',
                r'\bflame war\b',
                r'\bstirring up\b'
            ]
        }
        
        # Emotional intensity indicators
        self.intensity_patterns = {
            'high_emotion': [
                r'[!]{2,}',  # Multiple exclamation marks
                r'[?]{2,}',  # Multiple question marks
                r'[A-Z]{3,}',  # All caps words (3+ chars)
                r'[!?]{3,}',  # Mixed punctuation
                r'\b(VERY|EXTREMELY|ABSOLUTELY|TOTALLY)\b'
            ],
            'repetition': [
                r'(\w)\1{2,}',  # Repeated characters
                r'\b(\w+)\s+\1\b',  # Repeated words
                r'([.!?])\1{2,}'  # Repeated punctuation
            ]
        }
    
    def fit(self, X: List[str], y=None):
        """
        Fit the feature extractor (no fitting required for rule-based features).
        
        Args:
            X: List of text documents
            y: Target labels (unused)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X: List[str]) -> csr_matrix:
        """
        Transform text documents into domain-specific feature vectors.
        
        Args:
            X: List of text documents
            
        Returns:
            Sparse matrix of domain-specific features
        """
        features = []
        
        for text in X:
            text_features = self._extract_text_features(text)
            features.append(text_features)
        
        return csr_matrix(np.array(features))
    
    def _extract_text_features(self, text: str) -> List[float]:
        """
        Extract all domain-specific features from a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of feature values
        """
        text_lower = text.lower()
        features = []
        
        # 1. Banned word counts by category
        for category, words in self.banned_words.items():
            count = sum(text_lower.count(word) for word in words)
            features.append(count)
        
        # 2. Moderation term counts
        for category, terms in self.moderation_terms.items():
            count = sum(text_lower.count(term) for term in terms)
            features.append(count)
        
        # 3. Threat pattern detection
        threat_count = 0
        for pattern in self.threat_patterns:
            threat_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        features.append(threat_count)
        
        # 4. Harassment pattern detection
        harassment_count = 0
        for pattern in self.harassment_patterns:
            harassment_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        features.append(harassment_count)
        
        # 5. Politeness scoring
        positive_politeness = sum(text_lower.count(word) for word in self.politeness_indicators['positive'])
        negative_politeness = sum(text_lower.count(word) for word in self.politeness_indicators['negative'])
        politeness_score = positive_politeness - negative_politeness
        features.extend([positive_politeness, negative_politeness, politeness_score])
        
        # 6. Community-specific violation patterns
        for category, patterns in self.community_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text_lower, re.IGNORECASE))
            features.append(count)
        
        # 7. Emotional intensity features
        for category, patterns in self.intensity_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text, re.IGNORECASE))
            features.append(count)
        
        # 8. Advanced behavioral indicators
        features.extend(self._extract_behavioral_features(text, text_lower))
        
        # 9. Context-specific features
        features.extend(self._extract_context_features(text, text_lower))
        
        return features
    
    def _extract_behavioral_features(self, text: str, text_lower: str) -> List[float]:
        """
        Extract behavioral indicators that suggest rule violations.
        
        Args:
            text: Original text
            text_lower: Lowercase text
            
        Returns:
            List of behavioral feature values
        """
        features = []
        
        # Aggressive punctuation usage
        aggressive_punct = len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]{3,}', text))
        features.append(aggressive_punct)
        
        # Excessive capitalization (shouting)
        words = text.split()
        caps_ratio = sum(1 for word in words if word.isupper() and len(word) > 2) / len(words) if words else 0
        features.append(caps_ratio)
        
        # Personal attacks indicators
        personal_attacks = [
            'you are', 'you\'re', 'your', 'yourself'
        ]
        attack_count = 0
        for attack in personal_attacks:
            if attack in text_lower:
                # Check if followed by negative words
                pattern = f'{attack}\s+(stupid|dumb|idiot|moron|pathetic|worthless|useless)'
                attack_count += len(re.findall(pattern, text_lower))
        features.append(attack_count)
        
        # Dismissive language
        dismissive_phrases = [
            'whatever', 'don\'t care', 'who cares', 'so what', 'big deal',
            'get over it', 'move on', 'deal with it'
        ]
        dismissive_count = sum(text_lower.count(phrase) for phrase in dismissive_phrases)
        features.append(dismissive_count)
        
        # Authoritarian language
        authoritarian_phrases = [
            'you must', 'you should', 'you need to', 'you have to',
            'listen to me', 'do as I say', 'follow my advice'
        ]
        authoritarian_count = sum(text_lower.count(phrase) for phrase in authoritarian_phrases)
        features.append(authoritarian_count)
        
        return features
    
    def _extract_context_features(self, text: str, text_lower: str) -> List[float]:
        """
        Extract context-specific features for community moderation.
        
        Args:
            text: Original text
            text_lower: Lowercase text
            
        Returns:
            List of context feature values
        """
        features = []
        
        # Meta-discussion about moderation
        meta_moderation = [
            'this sub', 'this subreddit', 'this community', 'this forum',
            'the mods', 'the moderators', 'the rules', 'the guidelines'
        ]
        meta_count = sum(text_lower.count(phrase) for phrase in meta_moderation)
        features.append(meta_count)
        
        # Appeals and complaints
        appeal_phrases = [
            'unfair', 'unjust', 'biased', 'discrimination', 'censorship',
            'free speech', 'first amendment', 'my rights'
        ]
        appeal_count = sum(text_lower.count(phrase) for phrase in appeal_phrases)
        features.append(appeal_count)
        
        # Escalation indicators
        escalation_phrases = [
            'i\'ll report', 'i\'m reporting', 'contact admin', 'take this higher',
            'lawyer', 'legal action', 'sue', 'court'
        ]
        escalation_count = sum(text_lower.count(phrase) for phrase in escalation_phrases)
        features.append(escalation_count)
        
        # Community disruption
        disruption_phrases = [
            'ruin this', 'destroy this', 'burn it down', 'chaos', 'anarchy',
            'revolt', 'rebellion', 'uprising'
        ]
        disruption_count = sum(text_lower.count(phrase) for phrase in disruption_phrases)
        features.append(disruption_count)
        
        # Victim playing
        victim_phrases = [
            'always me', 'pick on me', 'target me', 'single me out',
            'everyone hates me', 'no one likes me', 'why me'
        ]
        victim_count = sum(text_lower.count(phrase) for phrase in victim_phrases)
        features.append(victim_count)
        
        # Manipulation tactics
        manipulation_phrases = [
            'if you don\'t', 'you owe me', 'after all i\'ve done',
            'guilt trip', 'emotional blackmail'
        ]
        manipulation_count = sum(text_lower.count(phrase) for phrase in manipulation_phrases)
        features.append(manipulation_count)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all domain-specific features.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Banned word categories
        for category in self.banned_words.keys():
            feature_names.append(f'banned_{category}_count')
        
        # Moderation terms
        for category in self.moderation_terms.keys():
            feature_names.append(f'mod_{category}_count')
        
        # Pattern-based features
        feature_names.extend([
            'threat_patterns', 'harassment_patterns',
            'positive_politeness', 'negative_politeness', 'politeness_score'
        ])
        
        # Community violation patterns
        for category in self.community_patterns.keys():
            feature_names.append(f'community_{category}_count')
        
        # Emotional intensity
        for category in self.intensity_patterns.keys():
            feature_names.append(f'intensity_{category}_count')
        
        # Behavioral features
        behavioral_names = [
            'aggressive_punctuation', 'caps_ratio', 'personal_attacks',
            'dismissive_language', 'authoritarian_language'
        ]
        feature_names.extend(behavioral_names)
        
        # Context features
        context_names = [
            'meta_moderation', 'appeals_complaints', 'escalation_indicators',
            'community_disruption', 'victim_playing', 'manipulation_tactics'
        ]
        feature_names.extend(context_names)
        
        return feature_names
    
    def fit_transform(self, X: List[str], y=None) -> csr_matrix:
        """
        Fit the extractor and transform the data.
        
        Args:
            X: List of text documents
            y: Target labels (unused)
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Provide detailed analysis of a single text for interpretability.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
            'text': text,
            'length': len(text),
            'word_count': len(text.split()),
            'violations': {},
            'risk_score': 0.0,
            'recommendations': []
        }
        
        text_lower = text.lower()
        
        # Analyze each category
        total_violations = 0
        
        # Check banned words
        for category, words in self.banned_words.items():
            count = sum(text_lower.count(word) for word in words)
            if count > 0:
                analysis['violations'][f'banned_{category}'] = count
                total_violations += count
        
        # Check threat patterns
        threat_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in self.threat_patterns)
        if threat_count > 0:
            analysis['violations']['threats'] = threat_count
            total_violations += threat_count * 3  # Weight threats more heavily
        
        # Check harassment patterns
        harassment_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in self.harassment_patterns)
        if harassment_count > 0:
            analysis['violations']['harassment'] = harassment_count
            total_violations += harassment_count * 2
        
        # Calculate risk score (0-1 scale)
        analysis['risk_score'] = min(total_violations / 10.0, 1.0)
        
        # Generate recommendations
        if analysis['risk_score'] > 0.7:
            analysis['recommendations'].append('High risk: Consider immediate moderation action')
        elif analysis['risk_score'] > 0.4:
            analysis['recommendations'].append('Medium risk: Monitor closely and consider warning')
        elif analysis['risk_score'] > 0.1:
            analysis['recommendations'].append('Low risk: Keep on watchlist')
        else:
            analysis['recommendations'].append('Low risk: No immediate action needed')
        
        return analysis