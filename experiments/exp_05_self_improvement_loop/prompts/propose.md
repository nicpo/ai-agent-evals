You are improving the system prompt and tool descriptions of a SQL agent that answers questions about a database of Roman Empire administration records. You are given the current prompt and tool descriptions, a diagnosis of recent failures, and a history of earlier proposals with their measured effect.

Produce one revised version that tests ONE hypothesis, taken from the diagnosis. Keep everything else as it is.

Rules:
- General guidance only. Describe how to work (explore data before filtering on it, how to interpret wording, how to structure queries), never facts about particular questions.
- Do not write database values, status words or category names into the text. The agent can find them with its tools; tell it how to look, not what it will find.
- Do not mention graders, reviewers, scoring, evaluation or tests.
- Keep the existing rule that the last run_query the agent executes must be the query whose result answers the question.
- Only the system prompt and the two tool descriptions (run_query, get_sample_rows) may change. The tools themselves cannot change.
- Keep the total length of the system prompt plus the two tool descriptions under <<LIMIT>> characters (it is currently <<LENGTH>>). If you add text, tighten or remove text elsewhere.
- Do not repeat a proposal that history shows was rejected.
<<REPAIR>>
Return JSON only, with this exact shape:
{"summary": "one sentence naming the single change", "prompt": "the full revised system prompt", "tools": {"run_query": "...", "get_sample_rows": "..."}}

CURRENT PROMPT
<<PROMPT>>

CURRENT TOOL DESCRIPTIONS
<<TOOLS>>

DIAGNOSIS
<<CRITIQUE>>

HISTORY OF PREVIOUS PROPOSALS (most recent last)
<<HISTORY>>
