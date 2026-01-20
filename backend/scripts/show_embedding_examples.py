#!/usr/bin/env python3
"""
Show example .env configurations for different embedding setups.
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           Embedding Provider Configuration Examples                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Choose one of these configurations based on your needs:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  LOCAL & FREE (Recommended for Privacy/Cost)
   Ollama nomic-embed-text (768D)

   DEFAULT_LLM_PROVIDER=ollama
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text

   ✅ Free, private, no API key needed
   ✅ 768D dimension (good storage/quality balance)
   ✅ Competitive quality with OpenAI
   ✅ Works offline

   Installation: ollama pull nomic-embed-text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2️⃣  PRODUCTION (Recommended for Quality)
   OpenAI text-embedding-3-small with dimension reduction (768D)

   DEFAULT_LLM_PROVIDER=openai
   DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
   OPENAI_API_KEY=sk-...
   EMBEDDING_DIMENSION=768

   ✅ High quality
   ✅ Same 768D dimension as Ollama (switch without re-indexing)
   ✅ Saves 50% storage vs default 1536D
   ✅ Minimal quality loss (<5%)

   Cost: $0.02 per 1M tokens

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3️⃣  MAXIMUM QUALITY (Production, High-Stakes)
   OpenAI text-embedding-3-large (3072D)

   DEFAULT_LLM_PROVIDER=openai
   DEFAULT_EMBEDDING_MODEL=text-embedding-3-large
   OPENAI_API_KEY=sk-...

   ✅ Best quality available
   ✅ 3072D dimension
   ⚠️  High storage cost (12GB per 1M embeddings)
   ⚠️  Slower search

   Cost: $0.13 per 1M tokens

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4️⃣  COST-OPTIMIZED PRODUCTION
   OpenAI text-embedding-3-small with aggressive reduction (512D)

   DEFAULT_LLM_PROVIDER=openai
   DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
   OPENAI_API_KEY=sk-...
   EMBEDDING_DIMENSION=512

   ✅ Saves 67% storage vs 1536D
   ✅ Minimal quality loss
   ✅ Faster search
   ⚠️  Cannot switch to Ollama without re-indexing

   Cost: $0.02 per 1M tokens

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5️⃣  DEVELOPMENT/TESTING (Fastest)
   HuggingFace all-MiniLM-L6-v2 (384D)

   DEFAULT_LLM_PROVIDER=huggingface
   DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

   ✅ Free, open source
   ✅ Very fast
   ✅ Smallest storage (384D)
   ⚠️  Lower quality than production models

   Installation: pip install sentence-transformers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6️⃣  LOCAL HIGH-QUALITY
   Ollama mxbai-embed-large (1024D)

   DEFAULT_LLM_PROVIDER=ollama
   OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

   ✅ Free, private
   ✅ Higher quality than nomic-embed-text
   ✅ 1024D dimension
   ⚠️  Larger model size (669MB)

   Installation: ollama pull mxbai-embed-large

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

7️⃣  MULTILINGUAL
   Cohere embed-multilingual-v3.0 (1024D)

   DEFAULT_LLM_PROVIDER=cohere
   COHERE_API_KEY=...

   ✅ Supports 100+ languages
   ✅ High quality
   ✅ 1024D dimension

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 RECOMMENDED SETUP (Dev + Prod Consistency)

   Development:
   ────────────
   DEFAULT_LLM_PROVIDER=ollama
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text

   Production:
   ───────────
   DEFAULT_LLM_PROVIDER=openai
   DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
   OPENAI_API_KEY=sk-...
   EMBEDDING_DIMENSION=768  # ← Match Ollama dimension

   ✅ Both use 768D - NO RE-INDEXING when deploying to production!
   ✅ Free in dev, pay in prod
   ✅ Smooth transition

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 DIMENSION COMPARISON

│ Dimension │ Storage  │ Speed     │ Quality       │ Use Case              │
├───────────┼──────────┼───────────┼───────────────┼───────────────────────┤
│ 384D      │ 1.5 GB   │ Very Fast │ Good          │ Dev/Testing           │
│ 512D      │ 2.0 GB   │ Fast      │ Very Good     │ Cost-Optimized Prod   │
│ 768D      │ 3.0 GB   │ Fast      │ Very Good     │ Balanced Prod         │
│ 1024D     │ 4.0 GB   │ Medium    │ Excellent     │ High-Quality Prod     │
│ 1536D     │ 6.0 GB   │ Medium    │ Excellent     │ OpenAI Default        │
│ 3072D     │ 12.0 GB  │ Slow      │ Best          │ Maximum Quality       │
└───────────┴──────────┴───────────┴───────────────┴───────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔄 SWITCHING PROVIDERS

Same Dimension (No Re-indexing):
  ✅ Ollama nomic-embed-text (768D) → OpenAI with EMBEDDING_DIMENSION=768
  ✅ OpenAI 768D → HuggingFace all-mpnet-base-v2 (768D)

Different Dimension (Re-indexing Required):
  ❌ Ollama nomic-embed-text (768D) → OpenAI default (1536D)
  ❌ OpenAI default (1536D) → HuggingFace all-MiniLM (384D)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 NEXT STEPS

1. Copy one of the configurations above to your .env file
2. Restart the application
3. Check configuration: python backend/scripts/check_embedding_dimension.py
4. Start indexing documents!

For more details, see:
- EMBEDDING_MODELS.md (complete model reference)
- EMBEDDING_DIMENSIONS.md (dimension guide)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
