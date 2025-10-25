# Ad Bias Detection - Multi-Agent System Architecture

## System Overview

This platform leverages Fetch.ai's multi-agent framework with ASI:ONE LLM integration to detect and analyze bias in advertising content (text, images, and videos). The system employs a distributed agent architecture where specialized agents work collaboratively to ingest, analyze, and score content for various forms of bias.

## Agent Architecture Diagram

```
User Input (Ad Content)
        ↓
┌───────────────────┐
│ Ingestion Agent   │
│ - Embeddings      │
│ - ChromaDB Store  │
└───────────────────┘
        ↓
    ┌───┴───┐
    ↓       ↓
┌────────┐ ┌────────┐
│ Text   │ │ Visual │
│ Bias   │ │ Bias   │
│ Agent  │ │ Agent  │
└────────┘ └────────┘
    ↓       ↓
    └───┬───┘
        ↓
┌───────────────────┐
│  Scoring Agent    │
│ - Aggregation     │
│ - Final Report    │
└───────────────────┘
```

---

## Agent Descriptions

### 1. Ingestion Agent
**Role:** Data Reception, Preprocessing, and Embedding Generation

**Description:**
The Ingestion Agent serves as the entry point for all ad content entering the system. It handles multi-modal data (text, images, videos) from the frontend and performs the following operations:

**Responsibilities:**
- Receive and validate incoming ad content from frontend API
- Extract and separate multi-modal components:
  - Text content (headlines, body copy, CTAs)
  - Visual content (images, video frames)
  - Metadata (target demographics, placement info)
- Generate embeddings using appropriate models:
  - Text: Sentence transformers (e.g., `all-MiniLM-L6-v2` or `text-embedding-ada-002`)
  - Images/Videos: Vision transformers (e.g., CLIP, ViT)
- Store embeddings in ChromaDB with metadata for retrieval
- Route content to specialized analysis agents
- Maintain data lineage and provenance tracking

**Tools/APIs:**
- `preprocess_content`: Cleans and normalizes input data
- `generate_text_embedding`: Creates text embeddings
- `generate_visual_embedding`: Creates visual embeddings
- `store_in_chromadb`: Persists embeddings with metadata
- `route_to_agents`: Dispatches content to analysis agents

**Output:**
- Structured content package with embeddings
- ChromaDB collection IDs
- Routing manifest for downstream agents

---

### 2. Text Bias Agent
**Role:** Text Content Analysis and Bias Detection

**Description:**
The Text Bias Agent is an expert system specialized in identifying linguistic bias patterns in advertising copy. It analyzes all textual elements for various forms of bias including gender, racial, age, socioeconomic, and cultural bias.

**Responsibilities:**
- Analyze textual content for bias indicators
- **RAG RETRIEVAL POINT #1**: Query ChromaDB for similar historical cases and bias patterns
- Identify specific bias types:
  - Gender bias (stereotyping, exclusionary language)
  - Racial/ethnic bias (cultural appropriation, stereotypes)
  - Age bias (ageism, generational stereotypes)
  - Socioeconomic bias (class assumptions)
  - Disability bias (ableist language)
  - LGBTQ+ bias (heteronormative assumptions)
- Extract specific problematic phrases/sentences
- Provide contextual explanation for each bias detection
- Maintain objectivity and avoid false positives

**Tools/APIs:**
- `analyze_text_bias`: LLM-powered bias detection
- `query_bias_knowledge_base`: RAG retrieval from ChromaDB
- `extract_problematic_phrases`: Highlights specific issues
- `classify_bias_type`: Categorizes detected bias
- `generate_text_report`: Creates structured findings

**Output:**
```json
{
  "agent": "text_bias_agent",
  "bias_detected": true,
  "bias_types": [
    {
      "type": "gender_bias",
      "severity": "medium",
      "examples": ["Only shows women in caregiving roles"],
      "context": "Reinforces traditional gender stereotypes",
      "confidence": 0.87
    }
  ],
  "overall_text_score": 6.2,
  "recommendations": ["Consider more diverse role representations"]
}
```

---

### 3. Visual Bias Agent
**Role:** Visual Content Analysis and Bias Detection

**Description:**
The Visual Bias Agent specializes in analyzing images and video content for visual representation bias. It examines composition, subject representation, context, and visual metaphors.

**Responsibilities:**
- Analyze visual content for representation bias
- **RAG RETRIEVAL POINT #2**: Query ChromaDB for similar visual patterns and problematic imagery
- Detect bias in:
  - Subject representation (diversity, tokenism)
  - Contextual placement (power dynamics, spatial positioning)
  - Color usage and symbolism
  - Body representation and stereotypes
  - Cultural symbols and appropriation
- Frame-by-frame analysis for video content
- Identify subtle visual cues and microaggressions
- Cross-reference with cultural sensitivity databases

**Tools/APIs:**
- `analyze_visual_bias`: Vision-LLM powered analysis
- `query_visual_patterns`: RAG retrieval from ChromaDB
- `detect_representation_metrics`: Measures diversity
- `analyze_composition`: Examines spatial dynamics
- `extract_video_keyframes`: Samples video content
- `generate_visual_report`: Creates structured findings

**Output:**
```json
{
  "agent": "visual_bias_agent",
  "bias_detected": true,
  "bias_types": [
    {
      "type": "representation_bias",
      "severity": "high",
      "examples": ["All leadership figures are white males"],
      "context": "Lacks diverse representation in authority roles",
      "confidence": 0.92
    }
  ],
  "diversity_metrics": {
    "gender_distribution": {"male": 0.8, "female": 0.2},
    "apparent_ethnicity": {"white": 0.9, "poc": 0.1}
  },
  "overall_visual_score": 4.5,
  "recommendations": ["Increase diverse representation in leadership roles"]
}
```

---

### 4. Scoring Agent
**Role:** Result Aggregation and Final Bias Assessment

**Description:**
The Scoring Agent serves as the final arbiter, synthesizing analyses from both Text and Visual Bias Agents to produce a comprehensive bias assessment score and actionable report.

**Responsibilities:**
- Receive and aggregate results from specialist agents
- Resolve conflicting assessments
- Calculate weighted bias scores:
  - Text bias weight: 40%
  - Visual bias weight: 40%
  - Intersectional bias weight: 20% (where text and visual biases compound)
- **RAG RETRIEVAL POINT #3**: Query ChromaDB for similar complete case studies to benchmark scoring
- Generate confidence intervals for scores
- Produce human-readable explanations
- Prioritize findings by severity and impact
- Generate actionable recommendations
- Create comparison reports (if historical data available)

**Tools/APIs:**
- `aggregate_agent_results`: Combines findings
- `calculate_weighted_score`: Computes final score
- `detect_intersectional_bias`: Identifies compounding biases
- `query_case_benchmarks`: RAG retrieval for comparative analysis
- `generate_recommendations`: Creates actionable guidance
- `create_final_report`: Produces comprehensive output

**Output:**
```json
{
  "agent": "scoring_agent",
  "overall_bias_score": 5.2,
  "score_breakdown": {
    "text_score": 6.2,
    "visual_score": 4.5,
    "intersectional_penalty": -0.3
  },
  "bias_summary": {
    "total_issues": 8,
    "high_severity": 2,
    "medium_severity": 4,
    "low_severity": 2
  },
  "top_concerns": [
    "Lack of diverse representation in leadership roles",
    "Gender stereotyping in role assignments"
  ],
  "recommendations": [
    {
      "priority": "high",
      "category": "representation",
      "action": "Include diverse individuals in positions of authority"
    }
  ],
  "confidence": 0.89,
  "similar_cases": ["case_123", "case_456"]
}
```

---

## RAG Retrieval Integration

### Where and Why RAG Happens

**RAG (Retrieval-Augmented Generation) occurs at THREE critical points:**

#### 1. **Text Bias Agent - Contextual Bias Detection**
- **When:** After initial text analysis, before final classification
- **What:** Queries ChromaDB for similar text patterns, historical bias cases, and linguistic patterns
- **Why:** 
  - Provides context from past detections
  - Reduces false positives by comparing to known patterns
  - Learns from historical decisions
  - Identifies emerging bias patterns
- **Query:** Uses text embeddings to find semantically similar content

#### 2. **Visual Bias Agent - Visual Pattern Recognition**
- **When:** During visual analysis, before generating report
- **What:** Queries ChromaDB for similar visual compositions, problematic imagery, and representation patterns
- **Why:**
  - Identifies subtle visual biases based on historical examples
  - Recognizes cultural symbols and their contexts
  - Learns from visual bias patterns
  - Provides evidence-based assessments
- **Query:** Uses visual embeddings (CLIP, ViT) to find similar images

#### 3. **Scoring Agent - Benchmarking and Calibration**
- **When:** After receiving both agent reports, during score calculation
- **What:** Queries ChromaDB for complete case studies with similar profiles
- **Why:**
  - Calibrates scoring against historical data
  - Ensures consistency across evaluations
  - Provides comparative context
  - Generates benchmark metrics
- **Query:** Uses combined embeddings or metadata filters

### ChromaDB Collection Structure

```python
# Suggested ChromaDB collections
collections = {
    "ad_content": {
        "embeddings": "text + visual combined",
        "metadata": ["timestamp", "source", "scores", "bias_types"]
    },
    "bias_patterns_text": {
        "embeddings": "text_embeddings",
        "metadata": ["bias_type", "severity", "examples", "context"]
    },
    "bias_patterns_visual": {
        "embeddings": "visual_embeddings",
        "metadata": ["bias_type", "severity", "visual_features", "context"]
    },
    "case_studies": {
        "embeddings": "combined_case_embeddings",
        "metadata": ["final_score", "bias_types", "recommendations", "outcome"]
    }
}
```

---

## Data Flow Sequence

```
1. User submits ad → Frontend API
2. Ingestion Agent receives content
3. Ingestion Agent generates embeddings
4. Ingestion Agent stores in ChromaDB (creates retrieval base)
5. Ingestion Agent dispatches to Text + Visual Agents (parallel)

-- PARALLEL EXECUTION --
6a. Text Bias Agent analyzes text
7a. Text Bias Agent queries ChromaDB (RAG) for similar patterns
8a. Text Bias Agent generates report

6b. Visual Bias Agent analyzes visuals
7b. Visual Bias Agent queries ChromaDB (RAG) for similar visuals
8b. Visual Bias Agent generates report
-- END PARALLEL --

9. Both agents send reports to Scoring Agent
10. Scoring Agent queries ChromaDB (RAG) for benchmarks
11. Scoring Agent aggregates and calculates final score
12. Scoring Agent generates comprehensive report
13. Report sent back to frontend
14. Results stored in ChromaDB for future RAG queries
```

---

## Additional Recommendations

### 1. Add a **Feedback Loop Agent** (Suggested Addition)
**Role:** Learning and Model Improvement

**Why:** 
- Collects user feedback on bias assessments
- Identifies false positives/negatives
- Continuously improves detection accuracy
- Updates ChromaDB with validated cases

**Integration:** Receives final report + user feedback → Updates ChromaDB knowledge base

### 2. Add a **Context Agent** (Suggested Addition)
**Role:** Cultural and Temporal Context Awareness

**Why:**
- Bias is contextual and culture-dependent
- What's acceptable varies by region, culture, time period
- Provides context-aware analysis

**Integration:** Runs before Text/Visual agents → Provides context parameters

### 3. Implement **Explainability Layer**
- Each agent should provide reasoning chains
- Store explanations in ChromaDB for transparency
- Use ASI:ONE's reasoning capabilities for interpretability

### 4. Consider **Real-time vs Batch Processing**
- Real-time: Single ad analysis (user upload)
- Batch: Campaign-level analysis (multiple ads)
- Different agent coordination strategies

### 5. **Bias Taxonomy Database**
- Create a comprehensive bias taxonomy
- Store in ChromaDB as a knowledge base
- Agents query this for classification consistency

---

## Technology Stack Recommendations

### Fetch.ai + ASI:ONE Integration
- Use ASI:ONE for LLM reasoning in all agents
- Fetch.ai uAgent framework for agent communication
- Agent protocols for inter-agent messaging

### Embedding Models
- **Text:** `sentence-transformers/all-mpnet-base-v2` or OpenAI `text-embedding-3-large`
- **Vision:** OpenAI `CLIP` or Google `ViT-L/14`
- **Multi-modal:** OpenAI `CLIP` for unified embedding space

### ChromaDB Configuration
- Use separate collections for different data types
- Implement hybrid search (semantic + metadata filtering)
- Set up periodic reindexing

### ASI:ONE LLM Prompting Strategy
- Use chain-of-thought prompting for bias detection
- Implement few-shot learning with bias examples
- Use structured output formatting (JSON mode)

---

## Next Steps

1. **Set up Fetch.ai uAgent framework** with ASI:ONE integration
2. **Initialize ChromaDB** with collections structure
3. **Implement Ingestion Agent** as the foundational piece
4. **Build Text and Visual Bias Agents** in parallel
5. **Develop Scoring Agent** with aggregation logic
6. **Create bias taxonomy** and seed ChromaDB
7. **Test with sample ads** and iterate
8. **Add feedback loop** for continuous improvement

---

## Scoring Scale

**Bias Score: 0-10**
- **0-3:** Significant bias detected (high concern)
- **4-6:** Moderate bias detected (needs revision)
- **7-8:** Minor bias detected (minor improvements)
- **9-10:** Minimal to no bias detected (approved)

---

*This architecture is designed to be scalable, maintainable, and continuously improving through RAG-based learning.*

