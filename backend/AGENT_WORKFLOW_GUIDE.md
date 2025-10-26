# 🤖 AGENT WORKFLOW GUIDE - Ad Bias Detection System

## 📋 Table of Contents
1. [Issues Fixed](#issues-fixed)
2. [Agent Interaction Flow](#agent-interaction-flow)
3. [Message Flow Diagram](#message-flow-diagram)
4. [How to Test](#how-to-test)
5. [Troubleshooting](#troubleshooting)

---

## ✅ Issues Fixed

### 1. **ChromaDB Dimension Mismatch** (FIXED ✅)
**Problem**: Text embeddings (384-dim) and visual embeddings (512-dim) were being stored in the same collection, causing errors.

**Solution**:
- Updated `backend/chroma.py` to create **separate collections**:
  - `adwhisper_text_embeddings` → 384-dimensional text embeddings
  - `adwhisper_visual_embeddings` → 512-dimensional visual embeddings
- Modified `ingestion_agent.py` to store embeddings in appropriate collections

**Files Changed**:
- `backend/chroma.py` ← Added `text_collection` and `visual_collection` properties
- `backend/agents/ingestion_agent.py:store_in_chromadb()` ← Stores in separate collections

---

### 2. **Visual Bias Agent Message Mismatch** (FIXED ✅)
**Problem**: Visual Bias Agent expected `VisualAnalysisRequest` but Ingestion Agent sent `EmbeddingPackage`.

**Solution**:
- Updated Visual Bias Agent to accept `EmbeddingPackage` (same as Text Bias Agent)
- Modified protocol handler to extract media URLs from metadata
- Added proper error handling and reporting

**Files Changed**:
- `backend/agents/visual_bias_agent.py` ← Changed protocol to accept `EmbeddingPackage`
- Added `send_to_scoring_agent()` function

---

### 3. **Scoring Agent Protocol Update** (FIXED ✅)
**Problem**: Scoring Agent expected `ScoringRequest` but bias agents send `BiasAnalysisComplete`.

**Solution**:
- Added handler for `BiasAnalysisComplete` messages
- Implemented state machine to wait for **both** Text and Visual reports
- Only generates final score when both reports are received

**Files Changed**:
- `backend/agents/scoring_agent.py` ← Added `handle_bias_analysis_complete()` handler
- Added `process_final_scoring()` to generate final reports

---

## 🔄 Agent Interaction Flow

### Step-by-Step Workflow

```
1️⃣ USER SUBMITS AD
   └─> FastAPI Backend receives YouTube URL

2️⃣ INGESTION AGENT
   ├─> Receives AdContentRequest from FastAPI
   ├─> Extracts YouTube transcript + thumbnail
   ├─> Generates embeddings:
   │   ├─> Text: 384-dim (all-MiniLM-L6-v2)
   │   └─> Visual: 512-dim (clip-ViT-B-32)
   ├─> Stores in ChromaDB (separate collections)
   └─> Sends EmbeddingPackage to:
       ├─> Text Bias Agent (port 8101)
       └─> Visual Bias Agent (port 8102)

3️⃣ TEXT BIAS AGENT (Parallel Processing)
   ├─> Receives EmbeddingPackage
   ├─> Analyzes text content
   ├─> RAG Query #1: ChromaDB text patterns
   ├─> Detects bias instances (gender, age, etc.)
   ├─> Calculates text score (0-10 scale)
   └─> Sends BiasAnalysisComplete to Scoring Agent

4️⃣ VISUAL BIAS AGENT (Parallel Processing)
   ├─> Receives EmbeddingPackage
   ├─> Analyzes visual content
   ├─> RAG Query #2: ChromaDB visual patterns
   ├─> Detects representation bias
   ├─> Calculates visual score (0-10 scale)
   └─> Sends BiasAnalysisComplete to Scoring Agent

5️⃣ SCORING AGENT (Waits for Both)
   ├─> Receives BiasAnalysisComplete from Text Agent
   ├─> Receives BiasAnalysisComplete from Visual Agent
   ├─> RAG Query #3: ChromaDB benchmark cases
   ├─> Detects intersectional bias
   ├─> Calculates weighted final score:
   │   └─> Text (40%) + Visual (40%) + Intersectional (20%)
   ├─> Aggregates all bias issues
   └─> Generates FinalBiasReport
       └─> (TODO: Send to FastAPI/Frontend)

6️⃣ FINAL REPORT
   └─> Contains:
       ├─> Overall score (0-10)
       ├─> Breakdown by category
       ├─> Top concerns
       ├─> Recommendations
       └─> Benchmark comparison
```

---

## 📨 Message Flow Diagram

```
┌─────────────────┐
│   FastAPI       │
│   Backend       │
└────────┬────────┘
         │ AdContentRequest
         ▼
┌─────────────────────┐
│ Ingestion Agent     │
│ (Port 8100)         │
│                     │
│ 1. Extract YouTube  │
│ 2. Generate Embeds  │
│ 3. Store ChromaDB   │
└──────┬──────┬───────┘
       │      │
       │      │ EmbeddingPackage (to both agents)
       │      │
   ┌───▼──────▼────┐
   │               │
┌──▼──────────┐ ┌─▼──────────────┐
│ Text Bias   │ │ Visual Bias    │
│ Agent       │ │ Agent          │
│ (Port 8101) │ │ (Port 8102)    │
│             │ │                │
│ RAG Query#1 │ │ RAG Query #2   │
└──────┬──────┘ └────────┬───────┘
       │                 │
       │ BiasAnalysisComplete
       │                 │
       └────────┬────────┘
                ▼
       ┌────────────────┐
       │ Scoring Agent  │
       │ (Port 8103)    │
       │                │
       │ 1. Wait both   │
       │ 2. RAG Query#3 │
       │ 3. Aggregate   │
       │ 4. Final Score │
       └────────────────┘
```

---

## 🧪 How to Test

### Prerequisites
Make sure all agents are running in separate terminals:

```bash
# Terminal 1 - Ingestion Agent
cd /Users/ronaldli/Desktop/Projects/calhacks/backend
./advenv/bin/python agents/ingestion_agent.py

# Terminal 2 - Text Bias Agent
./advenv/bin/python agents/text_bias_agent.py

# Terminal 3 - Visual Bias Agent
./advenv/bin/python agents/visual_bias_agent.py

# Terminal 4 - Scoring Agent
./advenv/bin/python agents/scoring_agent.py
```

### Test 1: Verify Agent Addresses Match

**IMPORTANT**: The agents have deterministic addresses based on their seeds. Check logs:

```
Text Bias Agent logs:
📍 Agent address: agent1q2f7k0hv7p63y9fjux702n68kyp3gdadljlfal4xpawylnxf2pvzjsppdlv

Visual Bias Agent logs:
📍 Agent address: agent1qtnatq0rhrj2pauyg2a8dgf56uqkf6tw3757z806w6c57zkw9nry2my2933

Scoring Agent logs:
📍 Agent address: agent1qv8q8vexn2l4hx08m30ecu329g0gfw3ede4ngf7j2gg756er4y5wcqlx9s8
```

**Verify these match the hardcoded addresses**:
- `ingestion_agent.py:86-87` → Text & Visual agent addresses
- `text_bias_agent.py:63` → Scoring agent address
- `visual_bias_agent.py:109` → Scoring agent address

### Test 2: Send Test Request

```bash
# Terminal 5 - Test the pipeline
cd /Users/ronaldli/Desktop/Projects/calhacks/backend
./advenv/bin/python verify_ingestion.py
```

### Test 3: Watch the Logs

Monitor all terminals to see the message flow:

**Expected Log Sequence**:

1. **Ingestion Agent** logs:
   ```
   📨 REST API: Received content request: <uuid>
   🔄 Preprocessing content...
   🎬 Detected YouTube URL, extracting content...
   ✅ Text embedding generated (384-dim)
   ✅ Visual embedding generated (512-dim)
   💾 Storing embeddings in ChromaDB...
   ✅ Text embedding stored (384-dim)
   ✅ Visual embedding stored (512-dim)
   📤 Routing to Text Bias Agent
   📤 Routing to Visual Bias Agent
   ```

2. **Text Bias Agent** logs:
   ```
   📨 Received content for text analysis: <uuid>
   🔍 Starting bias detection analysis...
   🔎 RAG RETRIEVAL: Querying ChromaDB...
   ✅ Found 2 similar cases
   📊 Calculating overall text bias score...
   📤 Sending results to Scoring Agent
   ```

3. **Visual Bias Agent** logs:
   ```
   📨 Received embedding package for visual analysis: <uuid>
   🔍 Extracting visual features...
   👁️ Analyzing visual content with Vision-LLM...
   🔎 RAG RETRIEVAL: Querying ChromaDB...
   📊 Calculating overall visual bias score...
   📤 Sending results to Scoring Agent
   ```

4. **Scoring Agent** logs:
   ```
   📨 Received text_bias_agent report for request: <uuid>
   ✅ Text bias report stored
   📊 Report status: Text=True, Visual=False
   ⏳ Waiting for remaining reports...

   📨 Received visual_bias_agent report for request: <uuid>
   ✅ Visual bias report stored
   📊 Report status: Text=True, Visual=True
   🎯 Both reports received! Generating final assessment...
   ⚖️ Calculating weighted final score...
   🎯 Final weighted score: 6.2
   ✅ Final report generated successfully!
   📊 Final Score: 6.2/10
   ```

---

## 🐛 Troubleshooting

### Issue: "No messages received by Text/Visual Bias Agents"

**Cause**: Agent addresses don't match

**Solution**:
1. Start all agents
2. Copy actual addresses from logs
3. Update hardcoded addresses in:
   - `ingestion_agent.py` lines 86-87
   - `text_bias_agent.py` line 63
   - `visual_bias_agent.py` line 109
4. Restart all agents

### Issue: "Collection expecting embedding with dimension of 384, got 512"

**Cause**: Old ChromaDB collection exists

**Solution**:
```bash
# Delete old ChromaDB data
rm -rf /Users/ronaldli/Desktop/Projects/calhacks/backend/.chroma_db

# Restart all agents
```

### Issue: "Scoring agent never generates final report"

**Cause**: Only one bias agent is sending results

**Solution**:
1. Check both Text and Visual agents are running
2. Check logs for errors in either agent
3. Verify agent addresses match

### Issue: "ChromaDB permission error"

**Cause**: ChromaDB lock file

**Solution**:
```bash
# Find and kill any Python processes holding the lock
pkill -f "python.*ingestion_agent"
pkill -f "python.*text_bias_agent"
pkill -f "python.*visual_bias_agent"
pkill -f "python.*scoring_agent"

# Restart agents
```

---

## 🎯 Next Steps

### 1. **Add FastAPI Integration**
The scoring agent currently just logs the final report. Need to:
- Add endpoint in FastAPI to receive final reports
- Send final report from Scoring Agent back to FastAPI
- Return to frontend

### 2. **Implement Actual LLM Analysis**
Currently using placeholder analysis. Need to integrate:
- **ASI:ONE** for text analysis (Text Bias Agent)
- **Vision-LLM** (GPT-4V or Claude Vision) for visual analysis

### 3. **Implement RAG Queries**
All three RAG retrieval points use placeholder data. Need to:
- Query ChromaDB text collection for similar text bias patterns
- Query ChromaDB visual collection for similar visual patterns
- Query ChromaDB for benchmark case studies

### 4. **Add Monitoring**
- Health check endpoints for each agent
- Metrics collection (processing time, success rate)
- Error tracking and alerting

---

## 📚 Reference

### Agent Ports
- **Ingestion Agent**: 8100
- **Text Bias Agent**: 8101
- **Visual Bias Agent**: 8102
- **Scoring Agent**: 8103

### Key Files
- `agents/shared_models.py` → All message models
- `agents/ingestion_agent.py` → Entry point
- `agents/text_bias_agent.py` → Text analysis
- `agents/visual_bias_agent.py` → Visual analysis
- `agents/scoring_agent.py` → Final aggregation
- `chroma.py` → ChromaDB interface

### Scoring Scale
- **0-3**: Significant bias detected (high concern)
- **4-6**: Moderate bias detected (needs revision)
- **7-8**: Minor bias detected (minor improvements)
- **9-10**: Minimal to no bias detected (approved)

---

**Created**: 2025-10-25
**Last Updated**: 2025-10-25
**Status**: ✅ All communication fixes applied
