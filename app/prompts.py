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
You MUST format your final answer **exactly** following the instructions below.
Do not add any extra text before or after the JSON.\n
\n
{format_instructions}
"""


#------------------------BUDGET---------------------------------------------------------
BUDGET_CHECK_PROMPT = """You are an expert budget evaluator for NACCER.

Your task is to assess whether the following research proposal COMPLIES with the specific BUDGET section of S&T (Science & Technology) Guidelines.

**CURRENT PROPOSAL TO EVALUATE:**
{proposal}

**BUDGET GUIDELINES:**
If required, on request, the date of receipt of funds by the implementing agency may be considered as
date of project commencement instead of commencement date of as mentioned in sanction letter, when
there is perceptible lag between these two events. Decision of Technical Sub-committee of SSRC
regarding project commencement date shall be treated as final.
12.0 DISBURSEMENT OF FUND TO PROJECTS
Fund shall only be disbursed to the Principal implementing agency. All the sub-implementing agencies
shall requisite fund through Principal implementing agency and funds will be disbursed to subimplementing agencies through Principal implementing agency except CMPDI & subsidiaries of CIL.
However, if any Private institute/organization carrying out the project along with subsidiaries of
CIL/SCCL/NLCIL, funds will be disbursed to subsidiaries of CIL/SCCL/NLCIL for further disbursement.
After approval of the project, the first installment of fund will be disbursed at one go by CMPDI within one
months of the receipt of the request in the prescribed form (Form-II). Request from the Principle
Implementing / Sub-implementing Agency(s) for disbursement of subsequent installment of fund for ongoing projects should also be submitted two months in advance along with the details of expenditure
already incurred and status of the progress of the project vis-à-vis approved work programme, in Forms
–III, IV and V, indicating how much fund is lying idle in their account which is liable for payment of interest.
Specific approval for any deviation regarding fund requirement / disbursement shall have to be obtained
by the Principal Implementing / Sub-implementing Agency(s) separately from the respective Competent
Authority.
Private institute/organization who are carrying out the project alongwith any Govt. Institute/organisation
shall submit fund requisition through the Institute/organisation concerned and it will be disbursed to
Private Research / Private Academic institutions/ organisations through concerned Institute/organisation
and/or any subsidiary of CIL, based on the progress of the project and as per their agreement with the
Private Research / Private Academic institutions/ organisations. In exceptional case, Technical subcommittee may instruct activity wise disbursement of fund to Private Research / Private Academic
institutions/ organisations directly.
Expenditure of fund for implementing the project activities will be allowed for the project duration only.
If presentation and acceptance of the completed project would not be done within the approved time
schedule of the project, Project proponent may separately indicate the expenditure for presentation,
within the approved project outlay (subject to celling of Travel limit) and get reimbursement of the same
accordingly. Decision of the concerned committee regarding the above expenditure shall be treated as
final.
13.0 MONITORING OF S&T PROJECTS
 Physical and financial status of all on-going S&T projects is to be monitored by CMPDI on regular
basis. Representatives of CMPDI may visit the sites/locale of project to assess the actual progress
of the project and to provide necessary guidance/instructions to complete the project within
approved time frame.

Salary and wages of the permanent employees of the Principal Implementing / Sub-implementing
Agency(s) are normally not admissible under S&T Grant. Though major scientific and technical work
is to be carried out by the Project Leader/Coordinator and co-investigators, some additional scientific
and technical personnel (JRFs/SRFs/RAs) may be engaged for working desired duration on the
project. Engagement of the personnel for implementation of the project shall be the responsibility of
the Principal Implementing or Sub-implementing Agency(s) as the case may be. The payment to be
made to the engaged personnel for the desired duration of the project shall be within the relevant
norms/Guidelines of the concerned related Principal Implementing or Sub-implementing Agency(s).
In case of engaging of technical personnel (JRFs/SRFs/RAs) in the research project, it is required to
engage SC and ST candidates in research projects as per the norms of the Government or at least
one person each from SC and ST category in each project. This is as per the Office Memorandum
dated 28.01.2019, issued by Under Secretary to the Govt. of India (Finance), MoC.
4.12 Justification for Manpower
Justification for number and level of staff to be engaged and also the resolution of the institution
regarding wages of the JRFs/SRFs/RAs.
4.13 Outlay - Contingency
Special requirements not covered under normal heads for any projects, may be indicated under this
section. Contingencies are meant to cover incidental expenditure and other miscellaneous expenditure
likely to be incurred during project implementation, due to increase in travel/ transport expenses /
changes of TA/DA, salary revision of research associates, porter charges and additional expenses
during preparation of project completion report etc. This should be limited to 5% of the total revenue
cost of the project.
4.14 Institute Overhead Cost with Justification
For meeting the cost of academic expense including infrastructural facilities, Institutional Overhead Cost
for academic/research institutes shall be within the provision mentioned as follows:
a) For projects costing up to Rs: 1.0 crore:Not more than 10% of the total cost for educational
institutions and 8% for laboratories and institutions under Central Government
Departments/Agencies except CSIR laboratories;
b) For projects costing more than Rs. 1.0 crore and upto Rs.5.0 Cr.:Rs. 15.0 lakh or 10% of Total
project cost, whichever is less:
c) For projects costing more than 5.0 Cr. and upto Rs. 20.0 Crore. Maximum Rs. 20.0 lakh will
be provided as overheads; and
d) For projects costing more than Rs. 20.0 Crore: the quantum will be decided on a case to case
basis and the decision of the approving committee will be treated as final.
The following norms will be applicable to individual centric Extra Mural Research (EMR) projects (i.e proposals
submitted by an individual researcher or a group of researchers students etc.) funded by the Ministry of Coal,
Government of India
(i) Equipment:
Full cost of equipment/s which is/are specifically recommended/approved by SSRC be provided
without discriminating the type of institutions (Public. Private. NGO etc.). Ownership of such equipment
shall be MoC and these shall not be disposed of without obtaining prior approval of the authority which
sanctioned the grant-in-aid. At the end of the project a specific request be made by the grantee
institution for retention/transfer of these equipment to the Institute subject to the institute in ensuring
proper upkeep of these equipment and making these available to the other researchers on
recommendation of MoC. A list of such equipment be placed in the website of the concerned institute
(ii) Emoluments:
The emoluments for manpower (other than JRF, SRF, RA and Research Scientist) to be fixed as per
the norms framed by DST.
(iii) Overheads:
Towards meeting the cost of academic expenses including infrastructural facilities, institute overhead
may be charged as mentioned below:
a) for projects costing upto Rs 1 crore, 10% of the total cost for educational institutions and NGOs and
8% for laboratories and institutions under Central Government Departments/Agencies;
b) for projects costing more than Rs 1.0 crore and upto Rs. 5.0 crore, overheads of Rs 15.0 lakh or
10% of total cost whichever is less,
c) for projects costing more than 5.0 crore and upto Rs 20.0 crore, Rs 20.0 lakh will be provided as
overheads, and
d) for projects costing more than Rs. 20.0 crore. the quantum will be decided on a case to case basis
(iv) Travel & Contingencies.
Maximum of Rs. 50,000/- each per annum may be provided for Travel and Contingencies. Higher
amount, based on the recommendations of the Expert Committee, to be provided where the research
work involves field work or/and project has many investigators / institutions and larger manpower. The
contingency amount may also be used for paying Registration fees for attending international
conferences.
(v) Consumables / Supplies & Materials:
The amount may be fixed by SSRC based on the recommendations of the Technical Sub-committee
of SSRC.

**INSTRUCTIONS:**
SECTION A: BUDGET LIMITS (Section 6.0)

1. Equipment Procurement
   ☐ Equipment solely for project work (not available at institution)
   ☐ Certificate from Head attached (Form 1A)
   ☐ Justification for each item provided
   ☐ No duplication of existing equipment
   Finding: [COMPLIANT / NON-COMPLIANT / NOT APPLICABLE]
   
2. Manpower
   ☐ Uses institution's permanent staff OR
   ☐ Additional JRF/SRF/RA engagement justified
   ☐ Payment within DST norms (reference provided)
   ☐ SC/ST engagement commitment included
   ☐ Form XI (Manpower details) attached
   Finding: [Status + specific issue if any]

3. Travel & Contingency Limits
   Budget Item        | Proposed | Limit         | Status
   -------------------|----------|---------------|--------
   Travel             | ₹XX,XXX  | ₹50k/yr/agency| ✓/✗
   Seminar/Workshop   | ₹XX,XXX  | ₹50k/agency   | ✓/✗
   Contingency        | ₹XX,XXX  | 5% of revenue | ✓/✗
   
   Finding: [Details]

4. TA/DA Budget
   ☐ ≤ ₹3 lakh per institute OR
   ☐ Excess justified with detailed calculations
   ☐ Form XII (Travel details) attached
   Finding: [Status]

5. Overhead Calculation
   Project Cost: ₹[XX] crore/lakh
   
   Applicable Rule:
   ☐ Up to ₹1 cr: 10% (edu/NGO) or 8% (govt lab)
   ☐ ₹1-5 cr: ₹15L or 10%, whichever less
   ☐ ₹5-20 cr: Max ₹20L
   ☐ >₹20 cr: Case-by-case
   
   Calculated: ₹[XX]
   Proposed: ₹[XX]
   Finding: [COMPLIANT / OVER / UNDER]

SECTION B: BUDGET JUSTIFICATION (Section 4.0, Annexure-I)

6. Equipment Justification (4.10)
   For each major equipment:
   ☐ Why needed for specific experiment/test
   ☐ Why indigenous model not suitable (if imported)
   ☐ Why not using existing equipment
   Finding: [Status]

7. Manpower Justification (4.12)
   ☐ Number and level justified
   ☐ Resolution regarding wages attached
   Finding: [Status]

8. Consumables Justification (6.0.v)
   ☐ Items identified
   ☐ Justification provided
   Finding: [Status]

SECTION C: REQUIRED DOCUMENTATION (Annexure-I, Section 2.0)

9. Supporting Forms
   ☐ Form-I (Main proposal)
   ☐ Form-IA (Head's endorsement)
   ☐ Form-IX (Equipment history - 7 years)
   ☐ Form-X (Computer history - 3 years)
   ☐ Form-XI (Manpower details)
   ☐ Form-XII (Travel details)
   ☐ Wage resolution documents
   ☐ Overhead policy documents
   Finding: [List missing forms]

SECTION D: PROHIBITED ITEMS (Section 7.0)

10. Items NOT Funded (unless specifically justified):
    ☐ Land/building (normally not funded)
    ☐ Permanent employee salaries
    ☐ Honorarium to existing employees
    ☐ Foreign travel
    ☐ Foreign expert fees (beyond proposal)
    ☐ Staff car
    ☐ Peons/attendants/stenographers
    ☐ Routine studies
    Finding: [Any prohibited items found?]

COMPLIANCE SUMMARY:

Budget Category        | Proposed (₹) | Compliant? | Issues
-----------------------|--------------|------------|--------
Equipment              | XX,XXX       | ✓/✗        | [Details]
Manpower               | XX,XXX       | ✓/✗        | [Details]
Consumables            | XX,XXX       | ✓/✗        | [Details]
Travel                 | XX,XXX       | ✓/✗        | [Details]
Seminar/Workshop       | XX,XXX       | ✓/✗        | [Details]
Contingency            | XX,XXX       | ✓/✗        | [Details]
Overhead               | XX,XXX       | ✓/✗        | [Details]
-----------------------|--------------|------------|--------
TOTAL BUDGET           | XX,XXX       |            |



═══════════════════════════════════════════════════════════════

OVERALL COMPLIANCE: [COMPLIANT / NEEDS REVISION / NON-COMPLIANT]

Compliance Score: [XX/100]
Scoring breakdown:
- Budget limits compliance: [XX/40]
- Justifications provided: [XX/30]
- Documentation complete: [XX/20]
- No prohibited items: [XX/10]

═══════════════════════════════════════════════════════════════

SPECIFIC RECOMMENDATIONS:

1. [Action item based on findings]
2. [Action item based on findings]
3. [Action item based on findings]

═══════════════════════════════════════════════════════════════


**YOUR ASSESSMENT:**
"""