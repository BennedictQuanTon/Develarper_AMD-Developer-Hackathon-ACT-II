import re

ROUTE_LOCAL_SENTIMENT = "LOCAL_SENTIMENT"
ROUTE_LOCAL_NER       = "LOCAL_NER"
ROUTE_LOCAL_GENERAL   = "LOCAL_GENERAL"   # Factual + short summaries
ROUTE_API_MATH        = "API_MATH"
ROUTE_API_CODE        = "API_CODE"
ROUTE_API_LOGIC       = "API_LOGIC"
ROUTE_API_LONG        = "API_LONG_CONTEXT" # Summarization > 6000 chars

THRESHOLD = 3  # Min score to avoid fallback to LOCAL_GENERAL

def classify(prompt: str) -> str:
    # 1. Fallback for large contexts to prevent local CPU OOM/Timeout
    if len(prompt) > 6000:
        return ROUTE_API_LONG

    p = prompt.lower()
    
    # Initialize score board
    scores = {
        ROUTE_API_MATH:        0,
        ROUTE_API_CODE:        0,
        ROUTE_API_LOGIC:       0,
        ROUTE_LOCAL_SENTIMENT: 0,
        ROUTE_LOCAL_NER:       0,
        ROUTE_LOCAL_GENERAL:   0,
    }

    # 2. Structural Signatures (High Confidence)
    # Match equations e.g., "45 * 12", "x = 5"
    if re.search(r'\d+\s*[\+\-\*\/\^=]\s*\d+', p):
        scores[ROUTE_API_MATH] += 5
        
    # Match markdown code blocks or common code structures
    if re.search(r'```|def \w+\(|class \w+:|\bpublic static\b|\bconsole\.log\b', prompt):
        scores[ROUTE_API_CODE] += 6
        
    # JSON schema requests
    if re.search(r'output.*json|format.*json|```json', p):
        scores[ROUTE_LOCAL_NER] += 2
        scores[ROUTE_LOCAL_SENTIMENT] += 1

    # 3. Weighted Keyword Dictionary (Using Word Boundaries \b to avoid partial matches)
    keywords = {
        ROUTE_API_MATH: {
            r'\bcalculate\b': 3, r'\bequation\b': 4, r'\bpercentage\b': 3,
            r'\bsolve for\b': 5, r'\bword problem\b': 4, r'\bderivative\b': 5,
            r'\bhow many\b': 2,  r'\bsum of\b': 3,    r'\bintegral\b': 5,
            r'\barithmetic\b': 4, r'\bmultiply\b': 3, r'\bdivide\b': 3,
        },
        ROUTE_API_CODE: {
            r'\bdebug\b': 4,    r'\bsyntax error\b': 5, r'\bcompile\b': 4,
            r'\brefactor\b': 4, r'\bscript\b': 3,       r'\bfunction\b': 2,
            r'\bgenerate.*code\b': 5, r'\bwrite.*program\b': 4,
            r'\bfix.*bug\b': 5, r'\bimport\b': 2,
            r'\bpython\b': 3,   r'\bcode\b': 2,         r'\bwrite a\b': 1,
            r'\bprogram\b': 2,
        },
        ROUTE_API_LOGIC: {
            r'\bconstraints?\b': 4, r'\bpuzzles?\b': 4,     r'\briddles?\b': 4,
            r'\bmust satisfy\b': 5, r'\blogic gate\b': 4, r'\bdeduce\b': 4,
            r'\bif.*then\b': 3,  r'\bonly if\b': 3,     r'\bsyllogism\b': 5,
            r'\bif\b': 1,        r'\bthan\b': 2,        r'\btaller\b': 2,
        },
        ROUTE_LOCAL_SENTIMENT: {
            r'\bsentiment\b': 5, r'\bpositive or negative\b': 5,
            r'\btone\b': 4,      r'\bemotion\b': 3,     r'\bfeel\b': 2,
            r'\bmood\b': 3,      r'\bhappy or sad\b': 4,
        },
        ROUTE_LOCAL_NER: {
            r'\bextract entities\b': 5, r'\bnamed entity\b': 5,
            r'\bperson.*org\b': 4,      r'\bidentify.*location\b': 4,
            r'\bwho.*mentioned\b': 3,   r'\bextract names\b': 5,
            r'\bnamed entit(y|ies)\b': 5, r'\bextract\b': 2,
            r'\bentit(y|ies)\b': 3,
        },
        ROUTE_LOCAL_GENERAL: {
            r'\bsummarize\b': 4, r'\bexplain\b': 2, r'\btldr\b': 4,
            r'\bwhat is\b': 2,   r'\bdefine\b': 2,  r'\bdescribe\b': 2,
            r'\bcapital of\b': 3, r'\bhistory of\b': 3, r'\btranslate\b': 3,
        }
    }

    # Apply weighted scores
    for category, weights in keywords.items():
        for pattern, weight in weights.items():
            if re.search(pattern, p):
                scores[category] += weight

    # 4. Negative Penalties (Anti-Patterns)
    # If it's a story or essay request, it's likely not a strict math/code problem
    if re.search(r'\bstory\b|\bpoem\b|\bessay\b|\bwrite a letter\b', p):
        scores[ROUTE_API_MATH]  -= 5
        scores[ROUTE_API_CODE]  -= 3
        scores[ROUTE_API_LOGIC] -= 5
        scores[ROUTE_LOCAL_GENERAL] += 3

    # 5. Determine the winner
    best_category = max(scores, key=scores.get)
    max_score = scores[best_category]

    if max_score < THRESHOLD:
        return ROUTE_LOCAL_GENERAL

    return best_category
