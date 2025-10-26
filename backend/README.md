# AdWhisper Backend - Multi-Agent Bias Detection System

A sophisticated multi-agent system for detecting bias in advertising content using Fetch.ai's uAgents framework with ASI:ONE LLM integration and RAG-powered analysis.

## Architecture Overview

The system consists of 4 specialized agents working collaboratively:

1. **Ingestion Agent** (Port 8100) - Receives ad content, generates embeddings, stores in ChromaDB
2. **Text Bias Agent** (Port 8101) - Analyzes text for linguistic bias using ASI:ONE + RAG
3. **Visual Bias Agent** (Port 8102) - Analyzes images/videos for visual bias + RAG
4. **Scoring Agent** (Port 8103) - Aggregates results, calculates final scores + RAG benchmarking

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Copy the environment template
cp env.example .env

# Edit .env and add your API keys
nano .env
```

### 3. Seed ChromaDB

Populate the database with initial bias patterns:

```bash
python seed_chromadb.py
```

Expected output:

```
✅ Seeded 10 text bias patterns
✅ Seeded 10 visual bias patterns
✅ Seeded 5 case studies
```

### 4. Run Individual Agents

Each agent runs independently. Open 4 separate terminals:

**Terminal 1 - Ingestion Agent:**

```bash
python agents/ingestion_agent.py
```

**Terminal 2 - Text Bias Agent:**

```bash
python agents/text_bias_agent.py
```

**Terminal 3 - Visual Bias Agent:**

```bash
python agents/visual_bias_agent.py
```

**Terminal 4 - Scoring Agent:**

```bash
python agents/scoring_agent.py
```

**Terminal 5 - FastAPI Server:**

```bash
python main.py
```

### 5. Get Agent Addresses

When each agent starts, it logs its address:

```
📍 Agent address: agent1qxxx...
```

Copy these addresses and update your `.env` file:

```
TEXT_BIAS_AGENT_ADDRESS=agent1q...
VISUAL_BIAS_AGENT_ADDRESS=agent1q...
SCORING_AGENT_ADDRESS=agent1q...
```

Also update the addresses in the respective agent files.

## API Usage

### Submit Ad for Analysis

```bash
curl -X POST http://localhost:8000/api/analyze-ad \
  -F "text_content=Looking for young rockstars to join our team of guys" \
  -F "image_url=https://example.com/ad-image.jpg"
```

Response:

```json
{
  "request_id": "uuid-here",
  "message": "Ad submitted successfully for bias analysis",
  "status": "processing",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Check Analysis Status

```bash
curl http://localhost:8000/api/status/{request_id}
```

### Get Results

```bash
curl http://localhost:8000/api/results/{request_id}
```

Response:

```json
{
  "request_id": "uuid",
  "status": "complete",
  "overall_score": 5.2,
  "assessment": "Moderate bias detected...",
  "bias_issues": [...],
  "recommendations": [...]
}
```

### Check System Health

```bash
curl http://localhost:8000/health
```

### View Collection Stats

```bash
curl http://localhost:8000/api/collections
```

## Agent Communication Flow

```
1. User → FastAPI → Ingestion Agent
2. Ingestion Agent → generates embeddings → stores in ChromaDB
3. Ingestion Agent → routes to → Text + Visual Agents (parallel)
4. Text Agent → RAG retrieval → analyzes text → sends report
5. Visual Agent → RAG retrieval → analyzes visuals → sends report
6. Scoring Agent → RAG benchmarking → aggregates → final report
7. Final report → stored in ChromaDB → returned to user
```

## RAG Retrieval Points

The system uses RAG (Retrieval-Augmented Generation) at 3 critical points:

1. **Text Bias Agent** - Queries `bias_patterns_text` collection for similar historical text bias cases
2. **Visual Bias Agent** - Queries `bias_patterns_visual` collection for similar visual bias patterns
3. **Scoring Agent** - Queries `case_studies` collection for benchmark comparison

## ChromaDB Collections

- `ad_content` - Complete ad submissions with embeddings
- `bias_patterns_text` - Historical text bias patterns (10+ seeded)
- `bias_patterns_visual` - Historical visual bias patterns (10+ seeded)
- `case_studies` - Complete analyzed cases for benchmarking (5+ seeded)

## Embeddings

- **Text**: Sentence-Transformers `all-MiniLM-L6-v2` (384-dim)
- **Visual**: OpenAI CLIP `ViT-B-32` (512-dim)

## Bias Detection Categories

### Text Bias

- Gender bias (gendered language, stereotypes)
- Racial/ethnic bias (coded language, stereotypes)
- Age bias (ageism, youth-centric language)
- Disability bias (ableist language)
- Socioeconomic bias (class assumptions)
- LGBTQ+ bias (heteronormative assumptions)

### Visual Bias

- Representation bias (lack of diversity)
- Contextual bias (power dynamics, positioning)
- Tokenism (superficial diversity)
- Body representation bias (beauty standards)
- Cultural appropriation
- Color symbolism bias

## Scoring Scale (0-10)

- **0-3**: Significant bias (high concern) - ❌
- **4-6**: Moderate bias (needs revision) - ⚠️
- **7-8**: Minor bias (minor improvements) - ⚡
- **9-10**: Minimal bias (approved) - ✅

## Development

### Project Structure

```
backend/
├── agents/
│   ├── ingestion_agent.py      # Entry point, embeddings, storage
│   ├── text_bias_agent.py      # Text analysis + RAG
│   ├── visual_bias_agent.py    # Visual analysis + RAG
│   └── scoring_agent.py        # Aggregation + RAG benchmarking
├── chroma.py                   # ChromaDB manager (4 collections)
├── main.py                     # FastAPI REST API
├── seed_chromadb.py           # Database seeding script
├── requirements.txt            # Python dependencies
├── env.example                 # Environment template
└── README.md                  # This file
```

### Adding New Bias Patterns

Edit `seed_chromadb.py` and add new patterns:

```python
text_bias_patterns.append({
    "id": "text_011",
    "text": "Your bias example text",
    "bias_type": "gender_bias",
    "severity": "high",
    "context": "Explanation",
    "examples": ["word1", "word2"]
})
```

Then re-run: `python seed_chromadb.py`

### Resetting Collections

```python
from chroma import ChromaDB
db = ChromaDB()
db.reset_collection(ChromaDB.COLLECTION_TEXT_PATTERNS)
```

## Troubleshooting

### Models Not Loading

```
Error: Could not load sentence-transformers model
```

Solution: Install with `pip install sentence-transformers torch`

### Agent Communication Failing

```
Error: Could not send message to agent
```

Solution:

1. Verify all agents are running
2. Check agent addresses in .env match actual addresses
3. Ensure ports are not blocked (8100-8103)

### ChromaDB Issues

```
Error: ChromaDB not initialized
```

Solution: Run `python seed_chromadb.py` first

### Rate Limiting (ASI:ONE)

```
Warning: ASI:ONE rate limit exceeded
```

Note: ASI:ONE agents are limited to 6 requests/hour. The system uses pattern-based analysis as fallback.

## Production Deployment

For production:

1. Use proper database (PostgreSQL/MongoDB) instead of in-memory request tracking
2. Set up Redis for caching and job queues
3. Deploy agents to Agentverse for 24/7 availability
4. Use environment variables for all configuration
5. Enable authentication/authorization on API endpoints
6. Set up monitoring and alerting
7. Use proper file storage (S3/GCS) for uploaded images/videos

## API Documentation

Interactive API docs available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or contributions:

- GitHub Issues: [Create an issue]
- Documentation: [Fetch.ai Innovation Lab](https://innovationlab.fetch.ai)

---

Built with ❤️ using Fetch.ai uAgents, ASI:ONE, ChromaDB, and FastAPI
