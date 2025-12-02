"""
Prompt templates for research proposal evaluation.
"""

NOVELTY_ANALYSIS_PROMPT = """You are an expert research evaluator for NACCER.

Your task is to carry out a NOVELTY ANALYSIS for the following research proposal.

**CURRENT PROPOSAL TO EVALUATE:**
{proposal}

**SIMILAR PROPOSALS SUBMITTED IN THE PAST:**
{context}

**INSTRUCTIONS:**
Analyze the novelty of the current proposal by comparing it with the similar past proposals above.

Provide your analysis covering:
1. Key similarities with past proposals
2. Novel aspects of the current proposal
3. Overall novelty assessment (High/Medium/Low)

**YOUR ANALYSIS:**
"""

COMPLIANCE_CHECK_PROMPT = """You are an expert research evaluator for NACCER.

Your task is to assess whether the following research proposal COMPLIES with the S&T (Science & Technology) Guidelines.

**CURRENT PROPOSAL TO EVALUATE:**
{proposal}

**RELEVANT S&T GUIDELINES:**
{context}

**INSTRUCTIONS:**
Carefully assess the proposal's compliance with each guideline provided above.

Provide your analysis covering:
1. Which guidelines the proposal COMPLIES with (include guideline number and reasoning)
2. Which guidelines the proposal DOES NOT COMPLY with (include guideline number and reasoning)
3. Overall compliance assessment (Fully Compliant/Partially Compliant/Non-Compliant)

**YOUR ASSESSMENT:**
"""

FINAL_EVALUATION_PROMPT = """You are an expert research evaluator for NACCER.

Your task is to carry out a DETAILED EVALUATION of the following research proposal.

**CURRENT PROPOSAL:**
{proposal}

**NOVELTY ASSESSMENT:**
{novelty}

**COMPLIANCE WITH S&T GUIDELINES:**
{compliance}

**INSTRUCTIONS:**
Based on the proposal, novelty assessment, and compliance analysis above, evaluate the proposal on the following aspects. For each aspect, provide a score out of 10 along with detailed reasoning.

Evaluate on these aspects:
1. **Budget** - Appropriateness and justification of budget allocation
2. **Technical Novelty** - Originality and innovation of the proposed research
3. **Technical Feasibility** - Practicality and achievability of the proposed methods
4. **Expertise** - Qualifications and capability of the research team
5. **Compliance with Guidelines** - Adherence to S&T guidelines and requirements
6. **Industry Relevance** - Practical applications and industrial impact
7. **Scalability** - Potential for scaling the solution
8. **Sustainability** - Long-term viability and environmental considerations
9. **Impact** - Overall potential impact on the field and society

For each aspect, provide:
- Score (X/10)
- Detailed reasoning

Finally, provide:
- **Overall Final Score** (Average of all aspects)
- **Summary** of the proposal's strengths and weaknesses

**YOUR DETAILED EVALUATION:**
"""

TALK2PROPOSAL_PROMPT = """You are an expert research proposal assistant for NACCER.

**QUESTION ASKED:**
{question}

**RELEVANT CHUNKS FROM THE REFERENCED PROPOSAL:**
{context}

**INSTRUCTIONS:**
- Answer the question STRICTLY based on the information provided in the relevant chunks above
- DO NOT answer anything outside of the provided chunks
- If the chunks do not contain enough information to answer the question, clearly state "The provided proposal sections do not contain sufficient information to answer this question"
- Be precise and reference specific details from the chunks when applicable
- Keep your answer focused and concise

**YOUR ANSWER:**
"""


SCORE_PROMPT = """
You are a strict evaluator.

Question:
{question}

Retrieved context:
{context}

Answer:
{answer}

Score from 0 to 100 how well the answer is supported by the context.
0 = not supported or wrong.
100 = fully supported and correct.
Return only the score as an integer in the 'score' field.
"""

