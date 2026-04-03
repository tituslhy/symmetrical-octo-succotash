import json
import time
from pathlib import Path

from tqdm import tqdm

from src.eval import ConversationResult, TurnResult, aggregate_results, evaluate_record
from src.logger import get_logger
from src.prompt import build_messages

logger = get_logger()

def save_results(results: list[ConversationResult], path: Path) -> None:
    """Serialize a list of conversation results to JSON"""
    serializable = []
    for r in results:
        serializable.append({
            "record_id": r.record_id,
            "turns": [
                {
                    "turn_index":    t.turn_index,
                    "question":      t.question,
                    "predicted":     t.predicted,
                    "predicted_raw": t.predicted_raw,
                    "gold_executed": t.gold_executed,
                    "gold_program":  t.gold_program,
                    "em":            t.em,
                    "num_match":     t.num_match,
                    "parse_error":   t.parse_error,
                }
                for t in r.turns
            ],
        })
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved {len(results)} results to {path}")


def load_results(path: Path) -> list[ConversationResult]:
    """Reloads previously saved results — skips re-running those records."""
    with open(path) as f:
        raw = json.load(f)

    results = []
    for r in raw:
        turns = [TurnResult(**t) for t in r["turns"]]
        cr = ConversationResult(record_id=r["record_id"])
        cr.turns = turns
        results.append(cr)
    return results


def run_pipeline(
    records:       list[dict],
    call_llm,
    output_path:   Path,
    mode:          str   = "gold",   # "gold" | "free"
    max_records:   int   = None,     # None = run all
    resume:        bool  = True,     # skip already-completed records
    sleep_between: float = 0.0,      # seconds between calls — useful for rate limits
) -> tuple[list["ConversationResult"], dict]:
    """
    Execute a full evaluation pipeline over a dataset of structured records.

    This function is designed as a resilient, resumable batch inference + evaluation loop
    for LLM-based pipelines. It supports checkpointing, partial reruns, rate limiting,
    and aggregation of evaluation metrics.

    Core responsibilities:
    - Filter out already-processed records (if resume is enabled)
    - Iterate through remaining records sequentially
    - Call an LLM-backed evaluation function per record
    - Persist intermediate results after each record (checkpointing)
    - Aggregate final evaluation metrics across all completed runs

    Inputs:
    - records: list of dictionaries, each containing at minimum:
        - "id": unique identifier (used for resume logic)
        - additional fields required by `evaluate_record`
    - call_llm: callable interface to an LLM (e.g., OpenAI client wrapper)
    - output_path: filesystem path for incremental result persistence
    - mode:
        - "gold": evaluation against reference/ground-truth answers
        - "free": open-ended or non-grounded evaluation mode
    - max_records: optional cap for partial runs (useful for debugging)
    - resume: if True, skips records already present in output_path
    - sleep_between: delay between LLM calls to avoid rate limiting

    Execution flow:
    1. Load previously completed results from `output_path` (if resume=True)
    2. Build set of completed record IDs
    3. Filter input records to only those not yet processed
    4. Optionally truncate to `max_records`
    5. Iterate over remaining records:
        a. Call `evaluate_record(...)`
        b. Append result to in-memory results list
        c. Persist full results to disk (checkpointing)
        d. Sleep if rate limiting is configured
    6. Aggregate all results into summary metrics
    7. Return full result list + metrics dictionary

    Failure handling:
    - Individual record failures do NOT stop the pipeline
    - Exceptions are caught per-record and logged
    - Failed records are skipped and pipeline continues

    Resumability guarantees:
    - Safe to interrupt (Ctrl+C) at any time
    - On restart with resume=True, already processed records are skipped
    - Progress is derived from saved output file, not in-memory state

    Output behavior:
    - results: list of ConversationResult objects (one per record)
    - metrics: aggregated summary statistics across all results
        (structure depends on `aggregate_results` implementation)

    Performance considerations:
    - checkpointing happens after every record (disk I/O heavy but safe)
    - sleep_between helps avoid API rate limits
    - max_records enables lightweight debugging runs

    LLM pipeline design intent:
    - deterministic replay of evaluation runs
    - fault-tolerant batch execution
    - incremental progress persistence for long-running jobs
    - compatibility with heterogeneous LLM backends via `call_llm`

    Returns:
        tuple:
            - results: list of per-record evaluation outputs
            - metrics: dictionary of aggregated evaluation statistics
    """
    output_path = Path(output_path)

    # resume: load whatever's already done
    completed_ids = set()
    results = []
    if resume and output_path.exists():
        results = load_results(output_path)
        completed_ids = {r.record_id for r in results}
        logger.info(f"Resuming — {len(completed_ids)} records already done")

    records_to_run = [
        r for r in records
        if r["id"] not in completed_ids
    ]
    if max_records:
        records_to_run = records_to_run[:max_records]

    logger.info(f"Running {len(records_to_run)} records in mode='{mode}'")

    for record in tqdm(records_to_run, desc=f"Evaluating [{mode}]"):
        try:
            result = evaluate_record(
                record            = record,
                call_llm          = call_llm,
                mode              = mode,
                build_messages_fn = build_messages,
            )
            results.append(result)

        except Exception as e:
            # don't let one bad record kill the whole run
            logger.warning(f"Failed on record {record['id']}: {e}")
            continue

        # checkpoint after every record — safe to Ctrl+C anytime
        save_results(results, output_path)

        if sleep_between:
            time.sleep(sleep_between)

    metrics = aggregate_results(results)
    return results, metrics
