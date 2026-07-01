# 🚀 Proposed Enhancements & Future Roadmap

## 1. Idea: Job Description Context Injection

**Problem**
Generated questions lack role-specific context, leading to generic interview preparation.

**Proposed Solution**
Allow users to input a job description so the system can tailor questions and answers based on role requirements.

**Implementation Direction**
- Parse job description into structured signals (skills, responsibilities, keywords)
- Inject extracted context into RAG prompt pipeline
- Optionally store embeddings of job descriptions for reuse

**Impact**
Improves relevance and aligns preparation with real-world interview expectations.

## 2. Idea: Dockerized Deployment

**Problem**
Current setup is not standardized, making deployment and environment setup inconsistent.

**Proposed Solution**
Add Docker support to containerize the application for consistent deployment.

**Implementation Direction**
- Create Dockerfile for backend services
- Add docker-compose for multi-service orchestration (API + vector DB)
- Optimize image size and dependency layers

**Impact**
Enables easy deployment, scalability, and reproducibility across environments.

## 3. Idea: Local Embedding Model Support

**Problem**
Dependency on external APIs for embeddings increases cost and latency.

**Proposed Solution**
Integrate support for local embedding models.

**Implementation Direction**
- Use models like sentence-transformers or Ollama embeddings
- Add abstraction layer to switch between local and API-based embeddings
- Optimize for batch processing

**Impact**
Reduces cost, improves privacy, and enables offline capability.

## 4. Idea: Company-Aware Question Intelligence via Web Scraping

**Problem**
System lacks awareness of company-specific interview trends and recent context.

**Proposed Solution**
Scrape company-related data (news, interview experiences) to generate more targeted questions.

**Implementation Direction**
- Build scraper for sources like company news, blogs, Glassdoor (if compliant)
- Extract trends and frequently asked topics
- Feed insights into RAG pipeline

**Impact**
Makes preparation highly targeted and increases real interview relevance.

## 5. Idea: Company Mapping & Role Alignment Tool

**Problem**
Users cannot map their preparation to specific companies and roles effectively.

**Proposed Solution**
Create a mapping system that aligns companies with expected question patterns and skills.

**Implementation Direction**
- Maintain structured database of companies → roles → question patterns
- Link with job description and scraped insights
- Enable filtering based on target company

**Impact**
Transforms the tool into a strategic interview preparation platform.

## 6. Idea: Adaptive Difficulty & Spaced Repetition

**Problem**
Questions are generated once with no feedback loop — users can't track which topics they're weak in or revisit them systematically.

**Proposed Solution**
Add a lightweight tracking layer that adjusts question difficulty based on user performance and resurfaces weak topics using spaced repetition.

**Implementation Direction**
- Tag each generated question with topic + difficulty metadata
- Store user response quality (self-rated or LLM-graded) per question
- Use a simple SM-2 style spaced repetition scheduler to resurface weak areas
- Gradually shift question difficulty up/down based on rolling performance

**Impact**
Turns the tool from a one-shot generator into a continuous prep loop, closer to real interview readiness.

## 7. Idea: Multi-Modal Answer Evaluation

**Problem**
Users can generate questions but have no way to check if their own answers are actually good.

**Proposed Solution**
Let users submit written or voice answers and get structured feedback (correctness, clarity, structure — e.g. STAR method for behavioral questions).

**Implementation Direction**
- Add answer input (text first, voice via Whisper later)
- Use RAG context + rubric prompt to evaluate against expected answer signals
- Return structured feedback: strengths, gaps, suggested improvements

**Impact**
Closes the loop from "generate questions" to "actually get better at answering them."

---

## Progress Tracker

| Idea | Status |
|---|---|
| Job Description Context Injection | 🔲 Not started |
| Dockerized Deployment | 🔲 Not started |
| Local Embedding Model Support | 🔲 Not started |
| Company-Aware Question Intelligence | 🔲 Not started |
| Company Mapping & Role Alignment | 🔲 Not started |
| Adaptive Difficulty & Spaced Repetition | 🔲 Not started |
| Multi-Modal Answer Evaluation | 🔲 Not started |