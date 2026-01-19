#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IDA Batch Runner v1.0
=====================
Runs IDA Pro in headless mode on multiple malware samples.
Processes 10,000+ samples with progress tracking and error logging.

Requirements:
- Python 3.8+
- tqdm (pip install tqdm)
- IDA Pro 7.x+ installed

Usage:
    python batch_runner.py --samples-dir ./malware --ida-path "C:/IDA/idat64.exe" --output-dir ./exports

Author: Antigravity Agent
License: MIT
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[!] tqdm not installed. Install with: pip install tqdm")


# === Configuration ===
@dataclass
class BatchConfig:
    """Batch processing configuration."""
    samples_dir: Path
    ida_path: Path
    output_dir: Path
    script_path: Path
    
    # Processing options
    max_workers: int = 4              # Parallel IDA instances
    timeout_seconds: int = 300        # 5 minutes per sample
    extensions: tuple = ('.exe', '.dll', '.sys', '.bin', '.so', '.elf', '.o')
    
    # Skip options
    skip_existing: bool = True        # Skip already processed samples
    skip_large_files_mb: int = 100    # Skip files larger than this
    
    # Logging
    log_file: Path = None
    error_log: Path = None


# === Logging Setup ===
def setup_logging(config: BatchConfig):
    """Configure logging for batch processing."""
    log_format = '%(asctime)s | %(levelname)s | %(message)s'
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    
    # File handler
    handlers = [console]
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger('batch_runner')


# === Sample Discovery ===
def discover_samples(config: BatchConfig, logger) -> List[Path]:
    """Find all malware samples in the directory."""
    samples = []
    
    logger.info(f"Scanning directory: {config.samples_dir}")
    
    for root, dirs, files in os.walk(config.samples_dir):
        for filename in files:
            filepath = Path(root) / filename
            
            # Check extension
            if not filename.lower().endswith(config.extensions):
                continue
            
            # Check file size
            try:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                if size_mb > config.skip_large_files_mb:
                    logger.debug(f"Skipping large file ({size_mb:.1f}MB): {filename}")
                    continue
            except:
                continue
            
            samples.append(filepath)
    
    logger.info(f"Found {len(samples)} samples")
    return samples


def get_sample_hash(filepath: Path) -> str:
    """Compute SHA256 hash of sample (for unique output naming)."""
    try:
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:16]  # First 16 chars
    except:
        return filepath.stem[:16]


def get_output_path(sample: Path, config: BatchConfig) -> Path:
    """Generate output path for a sample."""
    sample_hash = get_sample_hash(sample)
    output_name = f"{sample.stem}_{sample_hash}.ida_export.json"
    return config.output_dir / output_name


def get_processed_samples(config: BatchConfig) -> Set[str]:
    """Get set of already processed sample hashes."""
    processed = set()
    
    if not config.output_dir.exists():
        return processed
    
    for f in config.output_dir.glob("*.ida_export.json"):
        # Extract hash from filename: name_HASH.ida_export.json
        parts = f.stem.replace('.ida_export', '').rsplit('_', 1)
        if len(parts) == 2:
            processed.add(parts[1])
    
    return processed


# === IDA Execution ===
@dataclass
class ProcessResult:
    """Result of processing a single sample."""
    sample: Path
    output: Path
    success: bool
    duration: float
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


def run_ida_on_sample(sample: Path, config: BatchConfig, logger) -> ProcessResult:
    """Run IDA Pro on a single sample."""
    start_time = time.time()
    output_path = get_output_path(sample, config)
    
    # Build IDA command
    # -c: Create new database (disassemble from scratch)
    # -A: Autonomous mode (no dialogs)
    # -S: Run script
    # -L: Log file (optional)
    # -o: Output database path (optional, we don't need it)
    
    cmd = [
        str(config.ida_path),
        '-c',                                    # Create new database
        '-A',                                    # Autonomous/batch mode
        f'-S"{config.script_path}"',             # Run export script
        str(sample)
    ]
    
    # Set environment for the script
    env = os.environ.copy()
    env['IDA_BATCH_MODE'] = '1'
    env['IDA_EXPORT_OUTPUT'] = str(output_path)
    env['IDA_EXPORT_DIR'] = str(config.output_dir)
    
    try:
        logger.debug(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            shell=True  # Needed for Windows path handling
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0 and output_path.exists()
        
        return ProcessResult(
            sample=sample,
            output=output_path,
            success=success,
            duration=duration,
            error=None if success else f"Exit code: {result.returncode}",
            stdout=result.stdout,
            stderr=result.stderr
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.warning(f"Timeout: {sample.name}")
        return ProcessResult(
            sample=sample,
            output=output_path,
            success=False,
            duration=duration,
            error=f"Timeout after {config.timeout_seconds}s"
        )
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error: {sample.name}: {e}")
        return ProcessResult(
            sample=sample,
            output=output_path,
            success=False,
            duration=duration,
            error=str(e)
        )


# === Batch Processing ===
def process_batch(samples: List[Path], config: BatchConfig, logger) -> List[ProcessResult]:
    """Process all samples with parallel execution."""
    results = []
    
    # Filter already processed
    if config.skip_existing:
        processed_hashes = get_processed_samples(config)
        original_count = len(samples)
        samples = [s for s in samples if get_sample_hash(s) not in processed_hashes]
        skipped = original_count - len(samples)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed samples")
    
    if not samples:
        logger.info("No samples to process")
        return results
    
    logger.info(f"Processing {len(samples)} samples with {config.max_workers} workers")
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Progress bar
    if TQDM_AVAILABLE:
        pbar = tqdm(total=len(samples), desc="Processing", unit="sample")
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(run_ida_on_sample, sample, config, logger): sample
            for sample in samples
        }
        
        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result.success:
                    logger.debug(f"✓ {sample.name} ({result.duration:.1f}s)")
                else:
                    logger.warning(f"✗ {sample.name}: {result.error}")
                    
            except Exception as e:
                logger.error(f"Exception: {sample.name}: {e}")
                results.append(ProcessResult(
                    sample=sample,
                    output=get_output_path(sample, config),
                    success=False,
                    duration=0,
                    error=str(e)
                ))
            
            if TQDM_AVAILABLE:
                pbar.update(1)
    
    if TQDM_AVAILABLE:
        pbar.close()
    
    return results


def save_results_summary(results: List[ProcessResult], config: BatchConfig, logger):
    """Save processing summary."""
    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "total_duration_seconds": sum(r.duration for r in results),
        "avg_duration_seconds": sum(r.duration for r in results) / len(results) if results else 0,
        "failures": [
            {
                "sample": str(r.sample),
                "error": r.error
            }
            for r in results if not r.success
        ]
    }
    
    summary_path = config.output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Success: {summary['success']}/{summary['total']} ({100*summary['success']/summary['total'] if summary['total'] else 0:.1f}%)")
    
    # Save error log
    if config.error_log and summary['failures']:
        with open(config.error_log, 'w') as f:
            for failure in summary['failures']:
                f.write(f"{failure['sample']}\t{failure['error']}\n")
        logger.info(f"Error log saved to: {config.error_log}")


# === CLI ===
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run IDA Pro in batch mode on malware samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python batch_runner.py --samples-dir ./malware --ida-path "C:/IDA/idat64.exe"
    
    # With all options
    python batch_runner.py \\
        --samples-dir ./malware \\
        --ida-path "C:/Program Files/IDA Pro 8.3/idat64.exe" \\
        --output-dir ./exports \\
        --workers 8 \\
        --timeout 600
        """
    )
    
    parser.add_argument(
        '--samples-dir', '-s',
        type=Path,
        required=True,
        help="Directory containing malware samples"
    )
    
    parser.add_argument(
        '--ida-path', '-i',
        type=Path,
        required=True,
        help="Path to idat64.exe or idat.exe"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('./ida_exports'),
        help="Output directory for JSON exports (default: ./ida_exports)"
    )
    
    parser.add_argument(
        '--script',
        type=Path,
        default=None,
        help="Path to export script (default: export_ida_to_json.py in same dir)"
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help="Number of parallel IDA instances (default: 4)"
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=300,
        help="Timeout per sample in seconds (default: 300)"
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help="Don't skip already processed samples"
    )
    
    parser.add_argument(
        '--max-size-mb',
        type=int,
        default=100,
        help="Skip files larger than this (MB, default: 100)"
    )
    
    parser.add_argument(
        '--log',
        type=Path,
        default=None,
        help="Log file path"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate paths
    if not args.samples_dir.exists():
        print(f"Error: Samples directory not found: {args.samples_dir}")
        sys.exit(1)
    
    if not args.ida_path.exists():
        print(f"Error: IDA not found: {args.ida_path}")
        sys.exit(1)
    
    # Find export script
    script_path = args.script
    if not script_path:
        script_path = Path(__file__).parent / "export_ida_to_json.py"
    
    if not script_path.exists():
        print(f"Error: Export script not found: {script_path}")
        sys.exit(1)
    
    # Create config
    config = BatchConfig(
        samples_dir=args.samples_dir,
        ida_path=args.ida_path,
        output_dir=args.output_dir,
        script_path=script_path,
        max_workers=args.workers,
        timeout_seconds=args.timeout,
        skip_existing=not args.no_skip,
        skip_large_files_mb=args.max_size_mb,
        log_file=args.log or (args.output_dir / "batch_runner.log"),
        error_log=args.output_dir / "errors.log"
    )
    
    # Setup logging
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("  IDA Batch Runner v1.0")
    logger.info("=" * 60)
    logger.info(f"  Samples: {config.samples_dir}")
    logger.info(f"  IDA: {config.ida_path}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Workers: {config.max_workers}")
    
    # Discover samples
    samples = discover_samples(config, logger)
    
    if not samples:
        logger.warning("No samples found!")
        sys.exit(0)
    
    # Process
    start_time = time.time()
    results = process_batch(samples, config, logger)
    total_time = time.time() - start_time
    
    # Summary
    logger.info(f"\nTotal time: {total_time/60:.1f} minutes")
    save_results_summary(results, config, logger)
    
    # Exit code
    failed = sum(1 for r in results if not r.success)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
