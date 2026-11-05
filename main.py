"""Orchestrator: load data -> Data Scientist Agent -> Report Agent -> number check."""
import argparse

from config import MODEL_NAME, OUT
from data_loader import load_student_data
from data_scientist import run_data_scientist
from report_agent import write_report
from verifier import check_numbers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", choices=["por", "mat"], default="por")
    args = ap.parse_args()

    print(f"Loading '{args.course}' data...")
    df = load_student_data(args.course)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    print("Data Scientist Agent running...")
    results = run_data_scientist(df)

    print(f"Report Agent running ({MODEL_NAME})...")
    report = write_report(results)

    flagged = check_numbers(report, results)
    if flagged:
        print(f"WARNING: numbers in the report not found in results.json: {flagged}")
    else:
        print("Number check passed: all figures trace back to results.json")

    print(f"\nDone. See {OUT}/ for results.json, charts, and report.md\n")
    print(report)


if __name__ == "__main__":
    main()
