import json
import re
import sys

import typer
from rich import print as rich_print
from rich.panel import Panel

sys.path.append(".")

from src.llm_utils import stream_llm
from src.prompt import build_messages

app = typer.Typer(
    name="convfinqa",
    help="Conversational financial QA over ConvFinQA records",
    no_args_is_help=True,
)

def load_record(record_id: str, data_path: str = "data/convfinqa_dataset.json") -> dict:
    """Utility function to open dataset and find a specific record by ID."""
    with open(data_path) as f:
        data = json.load(f)
    for split in ["train", "dev", "test"]:
        for record in data.get(split, []):
            if record["id"] == record_id:
                return record
    raise ValueError(f"Record '{record_id}' not found in dataset")


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
    data_path: str = typer.Option("data/convfinqa_dataset.json", help="Path to dataset"),
) -> None:
    """
    Launch an interactive chat session over a single ConvFinQA record.

    This command enables multi-turn, freeform questioning over a financial
    document consisting of:
    - pre-table text
    - a structured table
    - post-table text

    The session maintains conversational state by accumulating prior
    questions and model answers, which are fed into subsequent prompts.

    Behavior:
    - Loads the specified record by `record_id`
    - Displays a short preview of the document
    - Enters a REPL-style loop for user input
    - Streams LLM responses token-by-token with visual formatting:
        - intermediate reasoning ("STEPS") shown in dim text
        - final answer ("ANSWER:") highlighted in blue

    Conversation flow:
    - Each user input is appended to `questions`
    - Model responses are parsed and appended to `answers`
    - `build_messages()` constructs the full prompt for each turn

    Example session:
    
        run `python main.py chat "Single_MO/2012/page_44.pdf-4"`

        >>> what was the cigars shipment volume change in 2012?
        ANSWER: -0.007

        >>> what about smokeable products overall?
        ANSWER: -0.040

        >>> what was the percentage change?
        ANSWER: -0.040

        >>> how does that compare to the prior year?
        ANSWER: 0.033

    Notes:
    - Later questions may depend on earlier answers (context is preserved)
    - The model emits both reasoning ("STEPS") and final outputs ("ANSWER:")
    - Only the parsed ANSWER is stored for future turns

    Exit conditions:
    - User types "exit" or "quit"
    - Empty input is ignored

    Error handling:
    - If the record is not found, exits with a message

    Side effects:
    - Prints streamed LLM output to stdout
    - Maintains in-memory conversation state only (no persistence)

    Args:
        record_id: Unique identifier of the dataset record
        data_path: Path to ConvFinQA dataset JSON file

    Returns:
        None (interactive CLI session)
    """
    try:
        record = load_record(record_id, data_path)
    except ValueError as e:
        rich_print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    doc = record["doc"]
    rich_print(Panel(
        f"[bold]Loaded:[/bold] {record_id}\n"
        f"[dim]Document preview: {doc['pre_text'][:120]}...[/dim]",
        title="ConvFinQA Chat",
        border_style="blue",
    ))
    rich_print("[dim]Type 'exit' or 'quit' to end the session.[/dim]\n")

    questions    = []
    answers      = []

    while True:
        message = input(">>> ").strip()

        if message.lower() in {"exit", "quit"}:
            rich_print("[dim]Session ended.[/dim]")
            break
        if not message:
            continue

        questions.append(message)

        messages = build_messages(
            pre_text      = doc["pre_text"],
            post_text     = doc["post_text"],
            table         = doc["table"],
            questions     = questions,
            prior_answers = answers,
            turn_index    = len(answers),
        )

        print("\n", end="", flush=True)

        full_response = []
        in_steps = True

        # Stream LLM response token-by-token with formatting
        for chunk in stream_llm(messages):
            full_response.append(chunk)
            
            # color the STEPS section in dim grey, brighten for ANSWER
            if "ANSWER:" in "".join(full_response):
                if in_steps:
                    # just crossed into ANSWER — switch colour
                    in_steps = False
                    sys.stdout.write("\033[0m")   # reset dim
                sys.stdout.write(f"\033[94m{chunk}\033[0m")  # bright blue
            else:
                sys.stdout.write(f"\033[2m{chunk}\033[0m")   # dim for STEPS
            
            sys.stdout.flush()

        print("\n", flush=True)

        raw    = "".join(full_response)
        match  = re.search(r"ANSWER:\s*(.+)", raw, re.IGNORECASE)
        answer = match.group(1).strip() if match else raw.strip()
        answers.append(answer)


@app.command()
def list_records(
    split:     str = typer.Option("dev", help="Split to list: train/dev/test"),
    data_path: str = typer.Option("data/convfinqa_dataset.json", help="Path to dataset"),
    limit:     int = typer.Option(20, help="Max records to show"),
) -> None:
    """
    List available record IDs from a specified dataset split.

    This is a utility command to quickly inspect the dataset and identify
    valid `record_id` values for use with the `chat` command.

    Behavior:
    - Loads dataset JSON from `data_path`
    - Selects records from the specified split (train/dev/test)
    - Displays up to `limit` records
    - For each record, prints:
        - record ID
        - number of conversational turns

    Output format:
        <record_id> (N turns)

    Assumptions:
    - Dataset follows ConvFinQA structure with:
        record["id"]
        record["dialogue"]["conv_questions"]

    Edge cases:
    - If split is not found, an empty list is shown
    - If limit exceeds available records, all records are displayed

    Args:
        split: Dataset split to inspect ("train", "dev", or "test")
        data_path: Path to dataset JSON file
        limit: Maximum number of records to display

    Returns:
        None (prints results to stdout)
    """
    with open(data_path) as f:
        data = json.load(f)

    records = data.get(split, [])[:limit]
    for r in records:
        turns = len(r["dialogue"]["conv_questions"])
        rich_print(f"[green]{r['id']}[/green] [dim]({turns} turns)[/dim]")


if __name__ == "__main__":
    app()
