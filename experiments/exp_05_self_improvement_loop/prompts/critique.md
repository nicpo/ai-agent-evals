You are analysing failures of a SQL agent that answers questions about a database of Roman Empire administration records. The agent has two tools, run_query and get_sample_rows, and a system prompt. You are shown the current prompt and tool descriptions, and several failed attempts. Each attempt lists the question, the agent's tool calls, its final SQL and answer, and a reviewer's verdict and reasoning.

Write a diagnosis of why these attempts failed:
1. Find the underlying behaviour or habit the failures have in common, not a per-question story.
2. For each failure, say whether a change to the agent's prompt or tool descriptions could plausibly have prevented it, or whether it comes from question ambiguity or reviewer strictness. Be willing to say a failure is not fixable by the prompt.
3. State the single most promising hypothesis for a prompt or tool-description change, in one or two sentences, in general terms.

Rules:
- The reviewer's reasoning may mention details of specific questions or database values. Do not repeat them. Describe behaviours (for example "filters on a guessed value without checking it"), not question content.
- Under 250 words.

CURRENT PROMPT
<<PROMPT>>

CURRENT TOOL DESCRIPTIONS
<<TOOLS>>

HISTORY OF PREVIOUS PROPOSALS (most recent last)
<<HISTORY>>

FAILED ATTEMPTS
<<FAILURES>>
