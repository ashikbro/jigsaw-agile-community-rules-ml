#!/usr/bin/env python3
"""
Bengali Dataset Generator for Content Moderation

This module generates balanced Bengali datasets for training content moderation models.
It includes both violation and safe examples with proper linguistic diversity.

Features:
1. Balanced violation/safe sample generation
2. Bengali linguistic patterns and variations
3. Cultural context awareness
4. Automatic data augmentation
5. Quality validation
6. Export to various formats

Author: AI Assistant
Date: 2025-01-31
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BengaliSample:
    """Bengali text sample with metadata."""
    text: str
    label: int  # 0 = safe, 1 = violation
    category: str
    subcategory: str
    severity: str  # low, medium, high
    cultural_context: str
    linguistic_features: List[str]

class BengaliDatasetGenerator:
    """Generator for balanced Bengali content moderation datasets."""
    
    def __init__(self, output_dir: str = "data/bengali_generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sample templates
        self._init_violation_templates()
        self._init_safe_templates()
        self._init_augmentation_patterns()
    
    def _init_violation_templates(self):
        """Initialize violation text templates."""
        self.violation_templates = {
            "harassment": {
                "personal_attack": [
                    "তুমি একটা {insult} এবং {negative_trait}",
                    "{person} একজন {insult}, {action} করা উচিত",
                    "তোমার মতো {insult} এই পৃথিবীতে থাকা উচিত নয়",
                    "{person} সত্যিই একটা {insult}, সবাই জানে",
                    "তুমি কতটা {negative_trait}, {insult} কোথাকার"
                ],
                "threats": [
                    "আমি তোমাকে {threat_action} করব",
                    "তোমার {body_part} {violent_action} করে দেব",
                    "তুমি যদি {condition} তাহলে {consequence}",
                    "{person} কে {threat_action} করা দরকার",
                    "আমি জানি তুমি কোথায় থাকো, {threat_action}"
                ],
                "hate_speech": [
                    "{group} রা সবাই {negative_trait}",
                    "এই {group} দের {action} করা উচিত",
                    "{group} রা আমাদের দেশের {problem}",
                    "সব {group} একই রকম {negative_trait}",
                    "{group} দের এখানে থাকার অধিকার নেই"
                ]
            },
            "toxicity": {
                "profanity": [
                    "তুমি একটা {profanity}",
                    "{profanity} কোথাকার, {action}",
                    "এই {profanity} গুলো {negative_action}",
                    "{profanity} এর মতো {behavior}",
                    "সব {profanity} একই রকম"
                ],
                "sexual_content": [
                    "তোমার {body_part} দিয়ে {sexual_action}",
                    "আমি তোমাকে {sexual_action} করতে চাই",
                    "তুমি {sexual_description} এর মতো",
                    "{person} এর সাথে {sexual_action} করব",
                    "তোমার {body_part} কত {sexual_adjective}"
                ]
            },
            "discrimination": {
                "religious": [
                    "{religion} রা সবাই {negative_trait}",
                    "এই {religion} দের {discriminatory_action}",
                    "{religion} ধর্ম একটা {negative_description}",
                    "সব {religion} একই রকম {stereotype}",
                    "{religion} দের এখানে জায়গা নেই"
                ],
                "ethnic": [
                    "{ethnicity} রা {negative_trait}",
                    "এই {ethnicity} দের {action} করা উচিত",
                    "{ethnicity} জাতি {negative_description}",
                    "সব {ethnicity} একই {stereotype}",
                    "{ethnicity} দের {discriminatory_treatment}"
                ]
            }
        }
        
        # Word banks for template filling
        self.violation_words = {
            "insult": ["বোকা", "গাধা", "পাগল", "নির্বোধ", "মূর্খ", "হারামি", "জানোয়ার", "কুত্তা"],
            "negative_trait": ["বদমাশ", "খারাপ", "নোংরা", "ঘৃণ্য", "বিকৃত", "অসভ্য", "নিচু", "জঘন্য"],
            "threat_action": ["মেরে ফেলব", "শেষ করে দেব", "ধ্বংস করব", "শাস্তি দেব", "ক্ষতি করব"],
            "violent_action": ["ভেঙে দেব", "কেটে দেব", "পিটিয়ে দেব", "মারব", "আঘাত করব"],
            "body_part": ["মুখ", "মাথা", "হাত", "পা", "গলা"],
            "group": ["মুসলিম", "হিন্দু", "খ্রিস্টান", "বৌদ্ধ", "বিদেশি", "শরণার্থী"],
            "profanity": ["শালা", "হারামজাদা", "কুত্তার বাচ্চা", "শুয়োরের বাচ্চা"],
            "sexual_action": ["ধর্ষণ করব", "যৌন নির্যাতন করব", "অশ্লীল কাজ করব"],
            "sexual_description": ["বেশ্যা", "রাণ্ডি", "খানকি"],
            "sexual_adjective": ["সুন্দর", "আকর্ষণীয়", "কামুক"],
            "religion": ["ইসলাম", "হিন্দু", "খ্রিস্টান", "বৌদ্ধ"],
            "ethnicity": ["বাঙালি", "বিহারি", "চাকমা", "সাঁওতাল", "রোহিঙ্গা"],
            "stereotype": ["চোর", "মিথ্যুক", "অলস", "নোংরা", "অশিক্ষিত"]
        }
    
    def _init_safe_templates(self):
        """Initialize safe text templates."""
        self.safe_templates = {
            "positive": {
                "compliments": [
                    "তুমি সত্যিই {positive_trait}",
                    "{person} একজন {good_quality} মানুষ",
                    "তোমার {skill} দেখে মুগ্ধ হলাম",
                    "তুমি {achievement} এর জন্য অভিনন্দন",
                    "{person} এর {positive_action} প্রশংসনীয়"
                ],
                "encouragement": [
                    "তুমি পারবে, {encouragement}",
                    "হাল ছেড়ো না, {motivation}",
                    "তোমার {strength} আছে",
                    "চেষ্টা করতে থাকো, {success_message}",
                    "তুমি {capability} রাখো"
                ]
            },
            "neutral": {
                "information": [
                    "আজকের আবহাওয়া {weather_condition}",
                    "{place} এ {event} হচ্ছে",
                    "নতুন {product} বাজারে এসেছে",
                    "{time} এ {activity} করব",
                    "{topic} সম্পর্কে জানতে চাই"
                ],
                "questions": [
                    "তুমি কেমন আছো?",
                    "{topic} সম্পর্কে কি জানো?",
                    "কোথায় {place} পাওয়া যায়?",
                    "কিভাবে {action} করতে হয়?",
                    "কখন {event} হবে?"
                ]
            },
            "cultural": {
                "festivals": [
                    "{festival} এর শুভেচ্ছা",
                    "আজ {festival} উপলক্ষে {activity}",
                    "{festival} এ {traditional_food} খাব",
                    "{festival} এর প্রস্তুতি নিচ্ছি",
                    "{festival} সবার জন্য আনন্দের"
                ],
                "traditions": [
                    "আমাদের {tradition} খুব সুন্দর",
                    "{cultural_practice} আমাদের ঐতিহ্য",
                    "বাংলার {cultural_element} বিখ্যাত",
                    "{traditional_art} শিখতে চাই",
                    "আমাদের {heritage} গর্বের"
                ]
            },
            "educational": {
                "learning": [
                    "{subject} পড়তে ভালো লাগে",
                    "নতুন {skill} শিখছি",
                    "{topic} সম্পর্কে আরো জানতে চাই",
                    "{course} এ ভর্তি হব",
                    "{knowledge_area} এ দক্ষতা বাড়াতে চাই"
                ],
                "sharing": [
                    "{fact} জানতে পেরে ভালো লাগল",
                    "{information} শেয়ার করছি",
                    "এই {tip} কাজে লাগবে",
                    "{advice} মেনে চলো",
                    "{lesson} থেকে শিক্ষা নিয়েছি"
                ]
            }
        }
        
        # Word banks for safe templates
        self.safe_words = {
            "positive_trait": ["ভালো", "সুন্দর", "দয়ালু", "বুদ্ধিমান", "সৎ", "পরিশ্রমী", "সাহায্যকারী"],
            "good_quality": ["ভালো", "সৎ", "দয়ালু", "সহায়ক", "বন্ধুত্বপূর্ণ", "বিনয়ী"],
            "skill": ["কাজ", "পড়াশোনা", "শিল্প", "গান", "নাচ", "রান্na", "লেখা"],
            "achievement": ["সফলতা", "পুরস্কার", "উন্নতি", "অগ্রগতি", "ভালো ফলাফল"],
            "positive_action": ["সাহায্য", "দয়া", "ভালো কাজ", "সেবা", "অবদান"],
            "encouragement": ["চেষ্টা করো", "হাল ছেড়ো না", "এগিয়ে যাও"],
            "motivation": ["সফল হবে", "পারবে", "এগিয়ে যাবে"],
            "strength": ["শক্তি", "সাহস", "দক্ষতা", "যোগ্যতা"],
            "capability": ["ক্ষমতা", "দক্ষতা", "যোগ্যতা", "সামর্থ্য"],
            "weather_condition": ["ভালো", "খারাপ", "রোদ", "বৃষ্টি", "ঠান্ডা"],
            "festival": ["ঈদ", "পূজা", "পহেলা বৈশাখ", "বিজয় দিবস", "স্বাধীনতা দিবস"],
            "tradition": ["সংস্কৃতি", "ঐতিহ্য", "উৎসব", "রীতিনীতি"],
            "cultural_element": ["ভাষা", "সাহিত্য", "সংগীত", "নৃত্য", "খাবার"],
            "subject": ["বাংলা", "ইংরেজি", "গণিত", "বিজ্ঞান", "ইতিহাস"],
            "skill": ["প্রোগ্রামিং", "রান্না", "গান", "নাচ", "লেখা"]
        }
    
    def _init_augmentation_patterns(self):
        """Initialize text augmentation patterns."""
        self.augmentation_patterns = {
            "spelling_variations": {
                "তুমি": ["তুমি", "তুই", "আপনি"],
                "আমি": ["আমি", "আমরা"],
                "করব": ["করব", "করবো", "করি", "করতে পারি"],
                "ভালো": ["ভালো", "ভাল", "সুন্দর", "চমৎকার"],
                "খারাপ": ["খারাপ", "বাজে", "নিকৃষ্ট", "জঘন্য"]
            },
            "punctuation_variations": [
                "", "।", "!", "?", "...", "!!!", "???"
            ],
            "case_variations": True,
            "number_substitutions": {
                "এক": ["১", "1", "এক"],
                "দুই": ["২", "2", "দুই"],
                "তিন": ["৩", "3", "তিন"]
            }
        }
    
    def generate_violation_samples(self, count: int = 500) -> List[BengaliSample]:
        """Generate violation samples."""
        samples = []
        
        for _ in range(count):
            # Choose random category and subcategory
            category = random.choice(list(self.violation_templates.keys()))
            subcategory = random.choice(list(self.violation_templates[category].keys()))
            
            # Choose random template
            template = random.choice(self.violation_templates[category][subcategory])
            
            # Fill template with random words
            filled_text = self._fill_template(template, self.violation_words)
            
            # Apply augmentation
            augmented_text = self._augment_text(filled_text)
            
            # Determine severity
            severity = self._determine_severity(category, subcategory, filled_text)
            
            sample = BengaliSample(
                text=augmented_text,
                label=1,  # violation
                category=category,
                subcategory=subcategory,
                severity=severity,
                cultural_context="bengali",
                linguistic_features=self._extract_linguistic_features(augmented_text)
            )
            
            samples.append(sample)
        
        return samples
    
    def generate_safe_samples(self, count: int = 500) -> List[BengaliSample]:
        """Generate safe samples."""
        samples = []
        
        for _ in range(count):
            # Choose random category and subcategory
            category = random.choice(list(self.safe_templates.keys()))
            subcategory = random.choice(list(self.safe_templates[category].keys()))
            
            # Choose random template
            template = random.choice(self.safe_templates[category][subcategory])
            
            # Fill template with random words
            filled_text = self._fill_template(template, self.safe_words)
            
            # Apply augmentation
            augmented_text = self._augment_text(filled_text)
            
            sample = BengaliSample(
                text=augmented_text,
                label=0,  # safe
                category=category,
                subcategory=subcategory,
                severity="none",
                cultural_context="bengali",
                linguistic_features=self._extract_linguistic_features(augmented_text)
            )
            
            samples.append(sample)
        
        return samples
    
    def _fill_template(self, template: str, word_bank: Dict[str, List[str]]) -> str:
        """Fill template with random words from word bank."""
        import re
        
        # Find all placeholders in template
        placeholders = re.findall(r'\{([^}]+)\}', template)
        
        filled_text = template
        for placeholder in placeholders:
            if placeholder in word_bank:
                replacement = random.choice(word_bank[placeholder])
                filled_text = filled_text.replace(f"{{{placeholder}}}", replacement)
            else:
                # If placeholder not found, use a generic replacement
                filled_text = filled_text.replace(f"{{{placeholder}}}", "কিছু")
        
        return filled_text
    
    def _augment_text(self, text: str) -> str:
        """Apply text augmentation."""
        # Apply spelling variations
        for original, variations in self.augmentation_patterns["spelling_variations"].items():
            if original in text:
                text = text.replace(original, random.choice(variations))
        
        # Add random punctuation
        if not text.endswith(tuple(self.augmentation_patterns["punctuation_variations"])):
            text += random.choice(self.augmentation_patterns["punctuation_variations"])
        
        return text
    
    def _determine_severity(self, category: str, subcategory: str, text: str) -> str:
        """Determine severity of violation."""
        high_severity_keywords = ["মেরে ফেলব", "ধর্ষণ", "হত্যা", "শেষ করে দেব"]
        medium_severity_keywords = ["বোকা", "গাধা", "হারামি", "মারব"]
        
        text_lower = text.lower()
        
        for keyword in high_severity_keywords:
            if keyword in text_lower:
                return "high"
        
        for keyword in medium_severity_keywords:
            if keyword in text_lower:
                return "medium"
        
        return "low"
    
    def _extract_linguistic_features(self, text: str) -> List[str]:
        """Extract linguistic features from text."""
        features = []
        
        # Check for Bengali script
        if any('\u0980' <= char <= '\u09FF' for char in text):
            features.append("bengali_script")
        
        # Check for question marks
        if '?' in text or '?' in text:
            features.append("question")
        
        # Check for exclamation
        if '!' in text or '!' in text:
            features.append("exclamation")
        
        # Check for formal/informal pronouns
        if 'আপনি' in text:
            features.append("formal")
        elif 'তুমি' in text or 'তুই' in text:
            features.append("informal")
        
        # Check for religious terms
        religious_terms = ['আল্লাহ', 'ভগবান', 'ঈশ্বর', 'বুদ্ধ']
        if any(term in text for term in religious_terms):
            features.append("religious")
        
        return features
    
    def generate_balanced_dataset(self, total_samples: int = 1000) -> List[BengaliSample]:
        """Generate balanced dataset with equal violation and safe samples."""
        violation_count = total_samples // 2
        safe_count = total_samples - violation_count
        
        logger.info(f"Generating {violation_count} violation samples...")
        violation_samples = self.generate_violation_samples(violation_count)
        
        logger.info(f"Generating {safe_count} safe samples...")
        safe_samples = self.generate_safe_samples(safe_count)
        
        all_samples = violation_samples + safe_samples
        random.shuffle(all_samples)
        
        return all_samples
    
    def export_to_csv(self, samples: List[BengaliSample], filename: str = "bengali_dataset.csv"):
        """Export samples to CSV format."""
        data = []
        for sample in samples:
            data.append({
                'comment_text': sample.text,
                'label': sample.label,
                'category': sample.category,
                'subcategory': sample.subcategory,
                'severity': sample.severity,
                'cultural_context': sample.cultural_context,
                'linguistic_features': ','.join(sample.linguistic_features)
            })
        
        df = pd.DataFrame(data)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Dataset exported to {output_path}")
        return output_path
    
    def export_to_json(self, samples: List[BengaliSample], filename: str = "bengali_dataset.json"):
        """Export samples to JSON format."""
        data = []
        for sample in samples:
            data.append({
                'text': sample.text,
                'label': sample.label,
                'category': sample.category,
                'subcategory': sample.subcategory,
                'severity': sample.severity,
                'cultural_context': sample.cultural_context,
                'linguistic_features': sample.linguistic_features
            })
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset exported to {output_path}")
        return output_path
    
    def validate_dataset(self, samples: List[BengaliSample]) -> Dict[str, Any]:
        """Validate dataset quality and balance."""
        total_samples = len(samples)
        violation_samples = sum(1 for s in samples if s.label == 1)
        safe_samples = total_samples - violation_samples
        
        # Category distribution
        category_dist = {}
        for sample in samples:
            category_dist[sample.category] = category_dist.get(sample.category, 0) + 1
        
        # Severity distribution (for violations only)
        severity_dist = {}
        for sample in samples:
            if sample.label == 1:
                severity_dist[sample.severity] = severity_dist.get(sample.severity, 0) + 1
        
        # Text length statistics
        text_lengths = [len(sample.text) for sample in samples]
        
        validation_report = {
            'total_samples': total_samples,
            'violation_samples': violation_samples,
            'safe_samples': safe_samples,
            'balance_ratio': safe_samples / max(violation_samples, 1),
            'category_distribution': category_dist,
            'severity_distribution': severity_dist,
            'text_length_stats': {
                'min': min(text_lengths),
                'max': max(text_lengths),
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths)
            }
        }
        
        return validation_report

def main():
    """Generate Bengali dataset for content moderation."""
    # Initialize generator
    generator = BengaliDatasetGenerator()
    
    # Generate balanced dataset
    print("Generating Bengali content moderation dataset...")
    samples = generator.generate_balanced_dataset(total_samples=2000)
    
    # Validate dataset
    validation_report = generator.validate_dataset(samples)
    print("\nDataset Validation Report:")
    print(f"Total samples: {validation_report['total_samples']}")
    print(f"Violation samples: {validation_report['violation_samples']}")
    print(f"Safe samples: {validation_report['safe_samples']}")
    print(f"Balance ratio: {validation_report['balance_ratio']:.2f}")
    
    print("\nCategory distribution:")
    for category, count in validation_report['category_distribution'].items():
        print(f"  {category}: {count}")
    
    print("\nSeverity distribution (violations only):")
    for severity, count in validation_report['severity_distribution'].items():
        print(f"  {severity}: {count}")
    
    # Export datasets
    csv_path = generator.export_to_csv(samples)
    json_path = generator.export_to_json(samples)
    
    print(f"\nDatasets exported:")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    
    # Show sample texts
    print("\nSample violation texts:")
    violation_samples = [s for s in samples if s.label == 1][:5]
    for i, sample in enumerate(violation_samples, 1):
        print(f"{i}. {sample.text} (Category: {sample.category}, Severity: {sample.severity})")
    
    print("\nSample safe texts:")
    safe_samples = [s for s in samples if s.label == 0][:5]
    for i, sample in enumerate(safe_samples, 1):
        print(f"{i}. {sample.text} (Category: {sample.category})")

if __name__ == "__main__":
    main()