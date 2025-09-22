#!/usr/bin/env python3
"""Convenience script for running comprehensive audit and temporal evaluation."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.audit_temporal import run_audit
from src.utils import configure_logging


def run_command(cmd: list, description: str) -> int:
    """Run a command and return its exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
    else:
        print(f"✅ Command completed successfully")
    
    return result.returncode


def main():
    """Run comprehensive audit and temporal evaluation."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive audit and temporal evaluation for cp-tox-fusion-mini"
    )
    
    # Data arguments
    parser.add_argument("--data", required=True, help="Path to data CSV/parquet file")
    parser.add_argument("--y", required=True, help="Target column name")
    parser.add_argument("--features", required=True, help="Feature column pattern (e.g., 'x_*')")
    parser.add_argument("--groups", help="Comma-separated group columns (e.g., 'batch_id,plate_id,cell_line')")
    parser.add_argument("--date-col", default="assay_date", help="Date column for temporal splits")
    
    # Model arguments
    parser.add_argument("--model-checkpoint", help="Path to model checkpoint (for future MC-Dropout)")
    parser.add_argument("--mc-dropout", type=int, default=30, help="Number of MC-Dropout passes")
    
    # Output arguments
    parser.add_argument("--out", default="artifacts", help="Output base directory")
    
    # Processing arguments
    parser.add_argument("--skip-audit", action="store_true", help="Skip data audit")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip enhanced evaluation")
    
    args = parser.parse_args()
    
    configure_logging()
    logger = logging.getLogger("cp_tox.run_audit_temporal")
    
    # Create output directories
    audit_dir = Path(args.out) / "audit"
    eval_dir = Path(args.out) / "evaluation"
    
    audit_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    exit_codes = []
    generated_artifacts = []
    
    # === RUN DATA AUDIT ===
    if not args.skip_audit:
        logger.info("Running data audit")
        
        audit_cmd = [
            sys.executable, "-m", "src.audit_temporal",
            "--data", args.data,
            "--y", args.y,
            "--features", args.features,
            "--out", str(audit_dir)
        ]
        
        if args.groups:
            audit_cmd.extend(["--groups", args.groups])
        
        exit_code = run_command(audit_cmd, "Data Audit")
        exit_codes.append(exit_code)
        
        if exit_code == 0:
            generated_artifacts.extend([
                audit_dir / "audit_results.json",
                audit_dir / "audit_report.md"
            ])
    
    # === RUN ENHANCED EVALUATION ===
    if not args.skip_evaluation:
        logger.info("Running enhanced evaluation")
        
        eval_cmd = [
            sys.executable, "-m", "src.error_analysis",
            "--temporal-eval",
            "--date-col", args.date_col,
            "--out", str(eval_dir)
        ]
        
        if args.mc_dropout:
            eval_cmd.extend(["--mc-dropout", str(args.mc_dropout)])
        
        if args.groups:
            eval_cmd.extend(["--groups"] + args.groups.split(","))
        
        exit_code = run_command(eval_cmd, "Enhanced Evaluation")
        exit_codes.append(exit_code)
        
        if exit_code == 0:
            # These files would be created in timestamped subdirectory
            logger.info("Enhanced evaluation artifacts saved to timestamped subdirectory")
    
    # === SUMMARY ===
    print(f"\n{'='*80}")
    print("AUDIT AND TEMPORAL EVALUATION SUMMARY")
    print('='*80)
    
    if not args.skip_audit:
        print(f"📊 Data Audit: {'✅ SUCCESS' if exit_codes[0] == 0 else '❌ FAILED'}")
        if exit_codes[0] == 0:
            print(f"   📁 Audit artifacts: {audit_dir}")
            print(f"   📋 Report: {audit_dir / 'audit_report.md'}")
            print(f"   📈 JSON results: {audit_dir / 'audit_results.json'}")
    
    if not args.skip_evaluation:
        eval_idx = 1 if not args.skip_audit else 0
        if len(exit_codes) > eval_idx:
            print(f"🔬 Enhanced Evaluation: {'✅ SUCCESS' if exit_codes[eval_idx] == 0 else '❌ FAILED'}")
            if exit_codes[eval_idx] == 0:
                print(f"   📁 Evaluation artifacts: {eval_dir}/eval_YYYYMMDD_HHMMSS/")
                print(f"   📊 Includes: slice metrics, temporal analysis, uncertainty plots")
    
    print(f"\n🎯 Overall Status: {'✅ ALL PASSED' if all(code == 0 for code in exit_codes) else '⚠️  SOME FAILED'}")
    
    # Return overall exit code
    return max(exit_codes) if exit_codes else 0


if __name__ == "__main__":
    sys.exit(main())