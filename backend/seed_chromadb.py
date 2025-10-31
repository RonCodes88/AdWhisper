"""
Seed ChromaDB with Initial Bias Patterns and Case Studies

This script populates the ChromaDB collections with example data to enable
RAG retrieval from the start. Run this once after setting up the database.

Usage:
    python seed_chromadb.py
"""

import sys
from sentence_transformers import SentenceTransformer
import open_clip
import torch
from PIL import Image
import numpy as np

from chroma import ChromaDB

# Initialize models
print("Loading embedding models...")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval()
print("✅ Models loaded\n")

# Initialize ChromaDB
print("Initializing ChromaDB...")
db = ChromaDB()
print(f"✅ ChromaDB initialized with {len(db._collections)} collections\n")


def generate_text_embedding(text: str):
    """Generate text embedding"""
    return text_model.encode(text, convert_to_numpy=True).tolist()


def generate_visual_embedding_placeholder():
    """Generate placeholder visual embedding"""
    # In production, use actual images
    return [0.1] * 512


# ==================== Seed Text Bias Patterns ====================

print("Seeding text bias patterns...")

text_bias_patterns = [
    {
        "id": "text_001",
        "text": "Looking for young, energetic team members to join our startup",
        "bias_type": "age_bias",
        "severity": "medium",
        "context": "Age-preferring language that discriminates against older candidates",
        "examples": ["young", "energetic"]
    },
    {
        "id": "text_002",
        "text": "Seeking rockstar developers to join our all-star team of guys",
        "bias_type": "gender_bias",
        "severity": "medium",
        "context": "Uses male-default language and gendered sports metaphors",
        "examples": ["rockstar", "guys"]
    },
    {
        "id": "text_003",
        "text": "Looking for articulate professionals from urban backgrounds",
        "bias_type": "racial_bias",
        "severity": "high",
        "context": "Contains coded racial language",
        "examples": ["articulate", "urban"]
    },
    {
        "id": "text_004",
        "text": "Digital natives wanted: must be tech-savvy and social media fluent",
        "bias_type": "age_bias",
        "severity": "medium",
        "context": "Age-coded language favoring younger generations",
        "examples": ["digital natives", "tech-savvy"]
    },
    {
        "id": "text_005",
        "text": "Seeking chairman for our board - strong leadership required",
        "bias_type": "gender_bias",
        "severity": "medium",
        "context": "Uses gendered leadership terminology",
        "examples": ["chairman"]
    },
    {
        "id": "text_006",
        "text": "Need someone to man the front desk on weekdays",
        "bias_type": "gender_bias",
        "severity": "low",
        "context": "Uses gendered verb",
        "examples": ["man the desk"]
    },
    {
        "id": "text_007",
        "text": "Looking for candidates with crazy attention to detail",
        "bias_type": "disability_bias",
        "severity": "medium",
        "context": "Uses ableist language",
        "examples": ["crazy"]
    },
    {
        "id": "text_008",
        "text": "Mature professionals need not apply - seeking fresh talent",
        "bias_type": "age_bias",
        "severity": "high",
        "context": "Directly discriminates against older workers",
        "examples": ["mature", "fresh talent"]
    },
    {
        "id": "text_009",
        "text": "Housewives wanted for part-time cleaning positions",
        "bias_type": "gender_bias",
        "severity": "high",
        "context": "Stereotypes women in domestic roles",
        "examples": ["housewives"]
    },
    {
        "id": "text_010",
        "text": "Looking for normal, well-adjusted team members",
        "bias_type": "disability_bias",
        "severity": "medium",
        "context": "Implicitly excludes neurodivergent individuals",
        "examples": ["normal", "well-adjusted"]
    }
]

for pattern in text_bias_patterns:
    embedding = generate_text_embedding(pattern["text"])
    db.store_text_bias_pattern(
        pattern_id=pattern["id"],
        embedding=embedding,
        bias_type=pattern["bias_type"],
        severity=pattern["severity"],
        context=pattern["context"],
        examples=pattern["examples"]
    )
    print(f"  ✓ Stored text pattern: {pattern['id']} ({pattern['bias_type']})")

print(f"✅ Seeded {len(text_bias_patterns)} text bias patterns\n")


# ==================== Seed Visual Bias Patterns ====================

print("Seeding visual bias patterns...")

visual_bias_patterns = [
    {
        "id": "visual_001",
        "bias_type": "representation_bias",
        "severity": "high",
        "visual_features": "All leadership positions shown with white males, women in support roles",
        "context": "Lack of diverse representation in authority positions"
    },
    {
        "id": "visual_002",
        "bias_type": "contextual_bias",
        "severity": "medium",
        "visual_features": "Men in foreground/center, women in background/periphery",
        "context": "Spatial hierarchy reinforces gender power dynamics"
    },
    {
        "id": "visual_003",
        "bias_type": "representation_bias",
        "severity": "high",
        "visual_features": "Only young people shown, no older adults",
        "context": "Age exclusion in representation"
    },
    {
        "id": "visual_004",
        "bias_type": "tokenism",
        "severity": "medium",
        "visual_features": "Single person of color among many white people",
        "context": "Token diversity representation"
    },
    {
        "id": "visual_005",
        "bias_type": "body_representation_bias",
        "severity": "medium",
        "visual_features": "Only thin, conventionally attractive body types shown",
        "context": "Narrow beauty standards, excludes diverse body types"
    },
    {
        "id": "visual_006",
        "bias_type": "contextual_bias",
        "severity": "high",
        "visual_features": "People of color shown in service roles, white people in professional roles",
        "context": "Racial stereotyping in role assignment"
    },
    {
        "id": "visual_007",
        "bias_type": "representation_bias",
        "severity": "medium",
        "visual_features": "All families shown with traditional heterosexual couples",
        "context": "Exclusion of LGBTQ+ families"
    },
    {
        "id": "visual_008",
        "bias_type": "contextual_bias",
        "severity": "medium",
        "visual_features": "Women shown with children/domestic settings, men in offices",
        "context": "Gender role stereotyping"
    },
    {
        "id": "visual_009",
        "bias_type": "representation_bias",
        "severity": "high",
        "visual_features": "No visible disabilities or assistive devices shown",
        "context": "Complete exclusion of people with disabilities"
    },
    {
        "id": "visual_010",
        "bias_type": "color_symbolism_bias",
        "severity": "low",
        "visual_features": "Traditional gendered color usage (pink for women, blue for men)",
        "context": "Reinforces gender stereotypes through color"
    }
]

for pattern in visual_bias_patterns:
    embedding = generate_visual_embedding_placeholder()
    db.store_visual_bias_pattern(
        pattern_id=pattern["id"],
        embedding=embedding,
        bias_type=pattern["bias_type"],
        severity=pattern["severity"],
        visual_features=pattern["visual_features"],
        context=pattern["context"]
    )
    print(f"  ✓ Stored visual pattern: {pattern['id']} ({pattern['bias_type']})")

print(f"✅ Seeded {len(visual_bias_patterns)} visual bias patterns\n")


# ==================== Seed Case Studies ====================

print("Seeding case studies...")

case_studies = [
    {
        "id": "case_001",
        "final_score": 4.2,
        "bias_types": ["gender_bias", "age_bias"],
        "recommendations": [
            "Use gender-neutral language",
            "Remove age preferences"
        ],
        "description": "Tech recruitment ad with significant bias"
    },
    {
        "id": "case_002",
        "final_score": 6.5,
        "bias_types": ["representation_bias"],
        "recommendations": [
            "Increase visual diversity"
        ],
        "description": "Healthcare ad with moderate representation issues"
    },
    {
        "id": "case_003",
        "final_score": 3.8,
        "bias_types": ["gender_bias", "racial_bias", "age_bias"],
        "recommendations": [
            "Complete content revision required",
            "Diversify representation",
            "Remove discriminatory language"
        ],
        "description": "Financial services ad with multiple severe biases"
    },
    {
        "id": "case_004",
        "final_score": 8.1,
        "bias_types": ["contextual_bias"],
        "recommendations": [
            "Minor adjustments to spatial positioning"
        ],
        "description": "Retail ad with minor contextual issues"
    },
    {
        "id": "case_005",
        "final_score": 5.5,
        "bias_types": ["age_bias", "disability_bias"],
        "recommendations": [
            "Remove age-coded language",
            "Include disability representation"
        ],
        "description": "Educational services ad with moderate bias"
    }
]

for case in case_studies:
    # Create combined embedding
    combined_embedding = [float(case["final_score"] / 10.0)] * 896
    
    db.store_case_study(
        case_id=case["id"],
        combined_embedding=combined_embedding,
        final_score=case["final_score"],
        bias_types=case["bias_types"],
        recommendations=case["recommendations"],
        metadata={"description": case["description"]}
    )
    print(f"  ✓ Stored case study: {case['id']} (score: {case['final_score']})")

print(f"✅ Seeded {len(case_studies)} case studies\n")


# ==================== Summary ====================

print("=" * 60)
print("SEED COMPLETE")
print("=" * 60)
print(f"Text bias patterns:   {db.get_collection_count(ChromaDB.COLLECTION_TEXT_PATTERNS)}")
print(f"Visual bias patterns: {db.get_collection_count(ChromaDB.COLLECTION_VISUAL_PATTERNS)}")
print(f"Case studies:         {db.get_collection_count(ChromaDB.COLLECTION_CASE_STUDIES)}")
print(f"Ad content:           {db.get_collection_count(ChromaDB.COLLECTION_AD_CONTENT)}")
print("=" * 60)
print("\n✅ ChromaDB is now ready for RAG retrieval!")

