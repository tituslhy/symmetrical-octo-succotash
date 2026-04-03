import re
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv
from langsmith import traceable

_ = load_dotenv(find_dotenv())

@dataclass
class TurnResult:
    """
    Evaluation result for a single Q/A turn in a conversation.

    Stores model output, ground truth, and evaluation signals
    used for both exact-match and numerical correctness checks.
    """

    turn_index: int
    question: str

    predicted_raw: str   # full raw LLM output
    predicted: str       # extracted ANSWER: value

    gold_executed: float | str  # executed/reference result
    gold_program: str           # reference program (if applicable)

    em: bool          # exact match between predicted and gold
    num_match: bool   # numeric match within tolerance
    parse_error: bool # True if ANSWER: extraction failed

@dataclass  
class ConversationResult:
    """
    Aggregated evaluation result for a full record (multi-turn conversation).

    Contains per-turn results and exposes convenience metrics
    for conversation-level performance tracking.
    """
    
    record_id:    str
    turns:        list[TurnResult] = field(default_factory=list)

    @property
    def exe_acc(self) -> float:
        """
        Execution accuracy over all turns.

        Defined as:
            fraction of turns where num_match == True
        """
        if not self.turns:
            return 0.0
        return sum(t.num_match for t in self.turns) / len(self.turns)

    @property
    def all_correct(self) -> bool:
        """
        Strict conversation success flag.

        True only if every turn satisfies num_match == True.
        """
        return all(t.num_match for t in self.turns)

def parse_answer(response: str) -> tuple[str, bool]:
    """
    Extract the value after ANSWER: from the LLM response.
    Returns (parsed_value, parse_error).
    """
    match = re.search(r"ANSWER:\s*(.+)", response, re.IGNORECASE)
    if not match:
        return "", True
    return match.group(1).strip(), False


def normalise_numeric(value: str) -> float | None:
    """
    Try to coerce a string to float.
    Handles: '0.14136', '14.1%', '25,587', '$206588', '-0.05'
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    v = str(value).strip()
    v = v.replace(",", "").replace("$", "").replace("%", "")
    
    # if it was a percentage string, convert to decimal
    if "%" in str(value):
        try:
            return float(v) / 100
        except ValueError:
            return None
    try:
        return float(v)
    except ValueError:
        return None


def numerical_match(predicted: str, gold: float | str, tol: float = 0.01) -> bool:
    """
    Returns True if predicted and gold are numerically within tol (relative).
    Falls back to exact string match for non-numeric golds.
    tol=0.01 means within 1% — generous enough for rounding differences.
    """
    pred_num = normalise_numeric(predicted)
    gold_num = normalise_numeric(gold)

    if pred_num is None or gold_num is None:
        # non-numeric: fall back to string match
        return str(predicted).strip().lower() == str(gold).strip().lower()

    if gold_num == 0:
        return pred_num == 0

    return abs(pred_num - gold_num) / abs(gold_num) < tol

def evaluate_record(
    record:        dict,
    call_llm,                        
    mode:          str = "gold",     # "gold" | "free"  
    build_messages_fn = None,        
) -> ConversationResult:
    """
    mode='gold'  — feed gold executed_answers forward (tests reasoning quality)
    mode='free'  — feed model's own predicted answers forward (tests real pipeline)
    """
    # Extract structured inputs
    doc      = record["doc"]
    dialogue = record["dialogue"]
    result   = ConversationResult(record_id=record["id"])

    prior_answers = []

    # Main evaluation loop (turn by turn)
    for i, question in enumerate(dialogue["conv_questions"]):
        
        # Proceed to build full prompt including:
        # - document context
        # - table
        # - previous Q/A history
        # - current question
        messages = build_messages_fn(
            pre_text      = doc["pre_text"],
            post_text     = doc["post_text"],
            table         = doc["table"],
            questions     = dialogue["conv_questions"],
            prior_answers = prior_answers,
            turn_index    = i,
        )

        # call LLM for current turn prediction with LangSmith tracing
        @traceable(
            name="convfinqa_turn",
            metadata={
                "record_id": record["id"],
                "turn_index": i,
                "mode": mode,
                "question": question,
                "gold": str(dialogue["executed_answers"][i]),
            }
        )
        def traced_call(msgs):
            """Just a light wrapper ontop of the call_llm function to add langsmith tracing"""
            return call_llm(msgs)
        
        raw = traced_call(messages)
        
        # extract structured answer + parse failure flag
        predicted, perr = parse_answer(raw)
        
        # ground truth signals
        gold_exec       = dialogue["executed_answers"][i] #exepected final answer
        gold_prog       = dialogue["turn_program"][i]     #reference program (debug/eval)

        turn = TurnResult(
            turn_index    = i,
            question      = question,
            predicted_raw = raw,
            predicted     = predicted,
            gold_executed = gold_exec,
            gold_program  = gold_prog,
            em            = str(predicted).strip() == str(gold_exec).strip(),
            num_match     = numerical_match(predicted, gold_exec),
            parse_error   = perr,
        )
        result.turns.append(turn)
        
        # -----------------------------
        # History update strategy
        # -----------------------------
        # This controls what the model "remembers" in next turn:
        #
        # mode = "gold":
        #   → oracle forcing (teacher-forcing style)
        #   → always feed correct answers forward
        #
        # mode = "free":
        #   → autoregressive rollout
        #   → feed model predictions forward (or fallback to gold if parsing fails)
        if mode == "gold":
            prior_answers.append(str(gold_exec))
        else: #"free"
            prior_answers.append(predicted if not perr else str(gold_exec))

    return result

def aggregate_results(results: list[ConversationResult]) -> dict:
    """Utility function to aggregate results for turns."""
    all_turns = [t for r in results for t in r.turns]
    
    # turn-level breakdown by position (reproduces paper's Figure 5)
    by_position = {}
    for t in all_turns:
        pos = t.turn_index
        if pos not in by_position:
            by_position[pos] = []
        by_position[pos].append(t.num_match)

    return {
        "exe_acc":            sum(t.num_match for t in all_turns) / len(all_turns),
        "conv_acc":           sum(r.all_correct for r in results) / len(results),
        "parse_error_rate":   sum(t.parse_error for t in all_turns) / len(all_turns),
        "total_turns":        len(all_turns),
        "total_conversations":len(results),
        "acc_by_turn_position": {
            pos: sum(v)/len(v) 
            for pos, v in sorted(by_position.items())
        },
    }
