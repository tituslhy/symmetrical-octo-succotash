SYSTEM_PROMPT = """You are a financial analyst assistant.
You will be given a financial document (text and a data table) and a
conversation. Your job is to answer the latest question accurately.

Rules:
- Use ONLY information from the document and prior conversation turns.
- For any calculation, show each step explicitly before your final answer.
- If you use a value from a prior turn's answer, say so explicitly.
- Express percentages as decimals (e.g. 14.1% → 0.141) unless asked otherwise.
- Round to 5 decimal places max.

You MUST respond in this exact format:

STEPS:
1. [what you are looking for and where you found it]
2. [any operation applied, written as: operation(a, b) = result]
...

ANSWER: [final value only, no units or commentary]"""

def format_table(table: dict) -> str:
    """
    Convert a nested dictionary representation of a table into a markdown-formatted string.

    This function assumes the input table follows a column-oriented structure:
        {
            column_name_1: {row_key_1: value, row_key_2: value, ...},
            column_name_2: {row_key_1: value, row_key_2: value, ...},
            ...
        }

    The output will be a GitHub-flavored markdown table with:
    - A header row containing column names
    - A divider row using '---'
    - One row per row_key
    - A leading "Row" column that displays row keys

    Example:
        Input:
            {
                "Revenue": {"2022": 100, "2023": 120},
                "Profit": {"2022": 30, "2023": 50}
            }

        Output:
            | Row | Revenue | Profit |
            |---|---|---|
            | 2022 | 100 | 30 |
            | 2023 | 120 | 50 |

    Behavior and assumptions:
    - The table is non-jagged: all columns are expected to share the same row keys.
      If keys are missing in a column, empty strings ("") will be used as fallback.
    - The order of columns is determined by insertion order of the dictionary.
    - The order of rows is derived from the first column encountered.
    - All values are converted to strings via `str()` before rendering.

    Edge cases:
    - If `table` is empty or None-like, an empty string is returned.
    - If columns exist but contain empty dictionaries, only header and divider are returned.

    Implementation notes for LLM/code generation:
    - Extract column names first, then derive row keys from the first column.
    - Construct header and divider separately before iterating rows.
    - Use `.get(row_key, "")` to safely handle missing values.
    - Join rows using newline characters for final output.

    Returns:
        A string containing the markdown-formatted table.
    """
    if not table:
        return ""

    columns = list(table.keys())
    row_keys = list(table[columns[0]].keys())

    # Build header
    header = "| Row | " + " | ".join(columns) + " |"
    divider = "|---|" + "|---|" * len(columns)

    rows = []
    for row_key in row_keys:
        vals = [str(table[col].get(row_key, "")) for col in columns]
        rows.append(f"| {row_key} | " + " | ".join(vals) + " |")

    return "\n".join([header, divider] + rows)


def format_history(
    questions: list[str],
    answers: list[str],
    turn_index: int
) -> str:
    """
    Construct a textual representation of prior conversation turns up to a given index.

    This function formats historical question-answer pairs into a structured,
    sequential transcript that can be consumed by an LLM for context retention.

    Format:
        Q1: <question_1>
        A1: <answer_1>
        Q2: <question_2>
        A2: <answer_2>
        ...

    Only turns strictly BEFORE `turn_index` are included.

    Example:
        questions = ["What is revenue?", "What is profit?"]
        answers = ["Revenue is...", "Profit is..."]
        turn_index = 2

        Output:
            Q1: What is revenue?
            A1: Revenue is...
            Q2: What is profit?
            A2: Profit is...

    Behavior and assumptions:
    - `questions` and `answers` are aligned lists:
        len(answers) == turn_index
        len(questions) >= turn_index
    - Each question at index i corresponds to answer at index i.
    - turn_index is 0-based (i.e., turn_index=0 means no prior history).

    Edge cases:
    - If `turn_index == 0`, returns a sentinel string: "No prior turns."
    - If lists are shorter than expected, this may raise IndexError (caller responsibility).

    Design intent:
    - Preserve conversational grounding for multi-turn reasoning.
    - Allow the model to reference prior computed answers explicitly.
    - Maintain simple, predictable formatting for prompt injection into LLMs.

    Implementation notes:
    - Iterate from 0 to turn_index - 1.
    - Append lines in strict Q/A alternating order.
    - Use 1-based numbering for readability (Q1, A1, ...).

    Returns:
        A newline-delimited string representing prior conversation turns.
    """
    if turn_index == 0:
        return "No prior turns."

    lines = []
    for i in range(turn_index):
        lines.append(f"Q{i+1}: {questions[i]}")
        lines.append(f"A{i+1}: {answers[i]}")
    return "\n".join(lines)

def build_user_prompt(
    pre_text: str,
    post_text: str,
    table: dict,
    questions: list[str],
    prior_answers: list[str],
    turn_index: int,
) -> str:
    """
    Assemble a structured, multi-section prompt for a single LLM interaction turn.

    This function composes all relevant context into a single prompt string,
    including:
    1. Document context (pre-table text, table, post-table text)
    2. Conversation history (prior Q&A turns)
    3. The current question to be answered

    The output is designed to:
    - Maximize LLM comprehension of structured + unstructured data
    - Preserve conversational continuity
    - Provide clear segmentation using markdown headers

    Prompt structure:

        ## Financial document

        ### Text (before table)
        <pre_text>

        ### Table
        <formatted_table>

        ### Text (after table)
        <post_text>

        ---

        ## Conversation so far
        <formatted_history>

        ---

        ## Current question (turn N)
        <current_question>

    Behavior and assumptions:
    - `format_table()` is used to render the table into markdown.
    - `format_history()` is used to construct prior turns.
    - `turn_index` determines:
        - which question is "current"
        - how many prior answers are included
    - `prior_answers` must contain exactly `turn_index` elements.

    Design considerations:
    - Section headers are intentionally verbose to guide LLM attention.
    - Separation markers (`---`) help prevent context blending.
    - The prompt is optimized for reasoning over financial/tabular data.

    Edge cases:
    - Empty table → rendered as empty string
    - No prior history → "No prior turns."
    - Missing or malformed inputs may propagate errors from helper functions

    Implementation notes:
    - Always compute `table_str` and `history_str` first.
    - Extract the current question using `questions[turn_index]`.
    - Use f-string templating for readability and maintainability.

    Returns:
        A fully formatted prompt string ready to be sent as a user message to an LLM.
    """
    table_str = format_table(table)
    history_str = format_history(questions, prior_answers, turn_index)
    current_q = questions[turn_index]

    return f"""## Financial document

### Text (before table)
{pre_text}

### Table
{table_str}

### Text (after table)
{post_text}

---

## Conversation so far
{history_str}

---

## Current question (turn {turn_index + 1})
{current_q}"""


def build_messages(
    pre_text: str,
    post_text: str,
    table: dict,
    questions: list[str],
    prior_answers: list[str],
    turn_index: int,
) -> list[dict]:
    """
    Return the messages list ready to pass to any OpenAI-compatible API.
    Keeps system prompt separate for models that support it.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                pre_text, post_text, table,
                questions, prior_answers, turn_index
            ),
        },
    ]
