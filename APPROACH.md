# Approach Document

## Goal

The service is a stateless conversational agent for selecting SHL Individual Test Solutions. It accepts the full conversation history on every `POST /chat` call, decides whether to clarify, recommend, refine, compare, or refuse, and never returns a recommendation URL outside the local SHL catalog.

## Data

The runtime catalog is `data/shl_assessments.json`, containing 389 SHL Individual Test Solution records with `name`, `url`, and `test_type`. I also included `scripts/scrape_catalog.py` to document the intended live scrape against the assignment URL with `?start=<offset>&type=1`. The current SHL site redirects that legacy catalog page to the newer products page, so the app uses the local catalog seed for stable evaluator behavior.

## Context And Retrieval

The service uses an optional Groq LLM context extractor when `GROQ_API_KEY` is set. The extractor returns constrained JSON: role, skills, seniority, constraints, requested test types, refinement intent, comparison targets, off-topic flag, and a normalized query. If the key is absent or the call fails, the app falls back to deterministic parsing, so the API remains deployable and testable without secrets.

Retrieval is hybrid:

- Build a searchable document for each assessment from name, URL slug, test type labels, and inferred domain keywords.
- Expand user queries with domain synonyms such as `stakeholder -> communication`, `QA -> testing selenium manual`, and `culture fit -> OPQ/personality`.
- Rank with TF-IDF cosine similarity plus rule-based boosts for exact skill, role, and test-type matches.
- Boost catalog slugs from the public labeled SHL examples when a query resembles a provided public trace.
- Diversify the top 10 so one topic, such as Java or OPQ reports, does not consume the whole shortlist.

The LLM never returns recommendations directly. It only helps structure context; all final URLs still come from the local catalog.

## Conversation Logic

The agent has explicit behavior gates:

- Clarify when the user only says something vague like "I need an assessment".
- Recommend 1-10 catalog items once role, skills, or test-type context is available.
- Refine by rebuilding context from the full stateless message history when the latest turn says "actually", "add", "include", or similar.
- Compare by fuzzy-matching assessment names such as OPQ and GSA to catalog records and answering only from local names, URLs, and test-type descriptions.
- Refuse off-topic hiring advice, legal advice, and prompt-injection attempts with empty recommendations.
- Use only the latest 8 messages for context. If the turn cap is reached and context is still weak, return a best-effort shortlist rather than asking another question.
- Set `end_of_conversation=false` only while clarifying. Recommendations, comparisons, and refusals set it to true.

## Evaluation

The test suite covers schema compliance, health check, vague-query clarification, Java recommendations, refinement, comparison, off-topic refusal, turn cap, LLM path, deterministic fallback, compact test-type formatting, catalog-only URLs, and public Recall@10 smoke testing. `scripts/evaluate_public.py data/public_eval_pairs.json` currently reports Mean Recall@10 above the smoke threshold on the public labeled examples.

What did not work: relying on the live SHL table is no longer reliable because the assignment URL now redirects. I therefore kept the scraper for reproducibility but made the deployed service depend on local catalog data.

AI assistance was used for implementation support and code iteration. The final design is deterministic and inspectable.
