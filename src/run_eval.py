import json
import random
import sys
from pathlib import Path

sys.path.append(".")

from src.llm_utils import call_llm
from src.logger import get_logger
from src.pipeline import run_pipeline

logger = get_logger(__name__)

SEED       = 42
N_TYPE1    = 14
N_TYPE2    = 6
DATA_PATH  = Path("data/convfinqa_dataset.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def stratified_sample(records: list[dict], n_type1: int, n_type2: int, seed: int) -> list[dict]:
    """This function stratifies the records into two types based on the presence of type2 questions, 
    then samples n_type1 and n_type2 records from each type."""
    rng    = random.Random(seed)
    type1  = [r for r in records if not r["features"]["has_type2_question"]]
    type2  = [r for r in records if r["features"]["has_type2_question"]]
    sample = rng.sample(type1, n_type1) + rng.sample(type2, n_type2)
    rng.shuffle(sample)
    return sample

def main():
    """This is the main entry point for the evaluation pipeline. It loads the dataset, samples records, 
    runs the pipeline in both "gold" and "free" modes, computes metrics, and saves results."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    sample = stratified_sample(data["dev"], N_TYPE1, N_TYPE2, SEED)

    logger.info(f"Sample: {len(sample)} records ({N_TYPE1} type1, {N_TYPE2} type2)")
    logger.info("Record IDs in sample:")  
    for r in sample:                        
        logger.info(f"  {r['id']}") 

    results_gold, metrics_gold = run_pipeline(
        records     = sample,
        call_llm    = call_llm,
        output_path = OUTPUT_DIR / "results_gold_20.json",
        mode        = "gold",
        resume      = True,
    )
    logger.info(f"GOLD  exe_acc={metrics_gold['exe_acc']:.2%}  conv_acc={metrics_gold['conv_acc']:.2%}")

    results_free, metrics_free = run_pipeline(
        records     = sample,
        call_llm    = call_llm,
        output_path = OUTPUT_DIR / "results_free_20.json",
        mode        = "free",
        resume      = True,
    )
    logger.info(f"FREE  exe_acc={metrics_free['exe_acc']:.2%}  conv_acc={metrics_free['conv_acc']:.2%}")

    gap = metrics_gold["exe_acc"] - metrics_free["exe_acc"]
    logger.info(f"Error propagation gap: {gap:.2%}")

    summary = {
        "seed":       SEED,
        "n_records":  len(sample),
        "n_type1":    N_TYPE1,
        "n_type2":    N_TYPE2,
        "record_ids": [r["id"] for r in sample],
        "gold":       metrics_gold,
        "free":       metrics_free,
        "gap":        gap,
    }
    with open(OUTPUT_DIR / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved eval_summary.json")

if __name__ == "__main__":
    main()
