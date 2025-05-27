#!/usr/bin/env python3

import argparse
import logging
import signal
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from parser import LogParser
from stats import LogStats
from ml_analyzer import MLAnalyzer
from reporter import LogReporter
from budget_guard import BudgetGuard

# Global variables for graceful shutdown
shutdown_requested = False
results_collected = []
reporter = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print("\n\n🛑 Shutdown requested (Ctrl+C). Finishing current analysis and generating report...")
    shutdown_requested = True

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def analyze_cloud_only_batch(ml_analyzer, log_lines_batch, reporter):
    """Analyze a batch of log lines using cloud-only analysis with async processing"""
    
    logger = logging.getLogger(__name__)
    
    # Create async tasks for all lines in batch with rate limiting
    tasks = []
    for i, (line, template_id) in enumerate(log_lines_batch):
        if shutdown_requested:
            break
        
        # Add small delay between requests to avoid rate limiting
        if i > 0:
            await asyncio.sleep(0.2)  # 200ms delay between requests
            
        # Create async task for cloud analysis
        task = asyncio.create_task(ml_analyzer.analyze_cloud_async(line))
        tasks.append((task, line, template_id))
    
    # Process results as they complete
    for task, line, template_id in tasks:
        if shutdown_requested:
            break
        try:
            probability = await task
            if probability >= ml_analyzer.threshold:
                logger.info(f"Cloud anomaly detected (prob={probability:.2f}): {line.strip()}")
                reporter.add_anomaly(line, template_id, probability)
        except Exception as e:
            logger.error(f"Error analyzing line {line.strip()}: {str(e)}")

async def cloud_only_mode(args, budget_guard, log_parser, log_stats, ml_analyzer, reporter):
    """Run analysis in cloud-only mode with async processing"""
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Running in CLOUD-ONLY mode with async processing")
    
    batch_size = 10  # Process lines in batches
    batch = []
    
    try:
        with open(args.input, 'r') as f:
            for line in f:
                if shutdown_requested:
                    break
                    
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse log line
                template_id, variables, timestamp = log_parser.parse(line)
                
                # Update statistics (for reporting purposes)
                log_stats.update(template_id, variables, timestamp)
                
                # Add to batch
                batch.append((line, template_id))
                
                # Process batch when it's full
                if len(batch) >= batch_size:
                    await analyze_cloud_only_batch(ml_analyzer, batch, reporter)
                    batch = []
                    
                    # Check budget after each batch
                    if not budget_guard.can_make_request():
                        logger.warning("💰 Budget limit reached, stopping analysis")
                        break
            
            # Process remaining lines in batch
            if batch and not shutdown_requested:
                await analyze_cloud_only_batch(ml_analyzer, batch, reporter)
                
    except Exception as e:
        logger.error(f"Error in cloud-only mode: {str(e)}")
        raise

def normal_mode(args, budget_guard, log_parser, log_stats, ml_analyzer, reporter):
    """Run analysis in normal mode (local + cloud fallback)"""
    
    logger = logging.getLogger(__name__)
    logger.info("🔄 Running in NORMAL mode (local + cloud fallback)")
    
    try:
        with open(args.input, 'r') as f:
            for line in f:
                if shutdown_requested:
                    logger.info("Shutdown requested, stopping processing...")
                    break
                    
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Parse log line
                template_id, variables, timestamp = log_parser.parse(line)
                
                # Update statistics
                log_stats.update(template_id, variables, timestamp)
                
                # Check for anomalies - pass the original line for keyword checking
                if log_stats.is_anomaly(template_id, log_line=line):
                    logger.info(f"Potential anomaly detected, performing ML analysis: {line.strip()}")
                    
                    # Perform ML analysis
                    probability = ml_analyzer.analyze(line, template_id)
                    
                    # Report findings if above threshold
                    if probability >= args.threshold:
                        logger.info(f"Confirmed anomaly (prob={probability:.2f}): {line.strip()}")
                        reporter.add_anomaly(line, template_id, probability)
                    else:
                        logger.debug(f"False alarm (prob={probability:.2f}): {line.strip()}")

    except Exception as e:
        logger.error(f"Error in normal mode: {str(e)}")
        raise

def adaptive_analysis_mode(args, ml_analyzer, reporter):
    """Run analysis using adaptive context-aware approach"""
    
    logger = logging.getLogger(__name__)
    logger.info("🧠 Running ADAPTIVE CONTEXT-AWARE analysis")
    logger.info("💡 Phase 1: Discovering log context, Phase 2: Adaptive analysis")
    
    try:
        # Read all log lines
        with open(args.input, 'r') as f:
            log_lines = [line.strip() for line in f if line.strip()]
        
        if not log_lines:
            logger.warning("No log lines found in file")
            return
        
        logger.info(f"Loaded {len(log_lines):,} log lines for analysis")
        
        # Configure ML analyzer
        config = {
            'local_model': 'google/gemma-2b',
            'budget': {'max_budget': 45.0}
        }
        ml_analyzer_adaptive = MLAnalyzer(config)
        
        # Run adaptive analysis with shutdown check
        def check_shutdown():
            return shutdown_requested
        
        results = ml_analyzer_adaptive.analyze_logs(log_lines, shutdown_check=check_shutdown)
        
        # Convert results to reporter format
        for anomaly in results['anomalies']:
            reporter.add_anomaly(
                anomaly['content'],
                anomaly.get('template_id', 'unknown'),
                anomaly['confidence']
            )
        
        # Print comprehensive summary
        reporter.print_analysis_summary(results)
        
        # Show context info
        context_info = ml_analyzer_adaptive.get_context_info()
        if context_info['context']:
            logger.info("✅ Context discovery successful!")
        else:
            logger.warning("⚠️ Context discovery failed, using fallback analysis")
            
    except Exception as e:
        logger.error(f"Error in adaptive analysis mode: {str(e)}")
        raise

def main():
    global reporter, shutdown_requested
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze logs for anomalies')
    parser.add_argument('--input', '-i', required=True, help='Input log file')
    parser.add_argument('--output', '-o', default='results.json', help='Output JSON file')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, help='Anomaly threshold (0-1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--cloud-only', action='store_true', help='Use only cloud analysis (faster with async)')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive context-aware analysis (recommended)')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('log_analyzer.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        reporter = LogReporter()

        # Check which mode to run
        if args.adaptive:
            # New adaptive context-aware mode
            logger.info("🧠 ADAPTIVE MODE: Context-aware analysis with Gemma-2B")
            logger.info("💡 This discovers log structure first, then adapts analysis accordingly")
            adaptive_analysis_mode(args, None, reporter)
            
        else:
            # Legacy modes
            budget_guard = BudgetGuard(max_budget=45.0)
            log_parser = LogParser()
            log_stats = LogStats()
            ml_analyzer = MLAnalyzer(budget_guard, threshold=args.threshold)

            # Print mode information
            if args.cloud_only:
                logger.info("🌩️  CLOUD-ONLY MODE: Using async API requests for maximum speed")
                logger.info("💡 This mode skips local analysis and uses parallel cloud processing")
                asyncio.run(cloud_only_mode(args, budget_guard, log_parser, log_stats, ml_analyzer, reporter))
            else:
                logger.info("🏠 NORMAL MODE: Local analysis with cloud fallback")
                logger.info("💡 Use --adaptive for smarter context-aware analysis")
                normal_mode(args, budget_guard, log_parser, log_stats, ml_analyzer, reporter)

        # Generate report (this will run even if Ctrl+C was pressed)
        if shutdown_requested:
            logger.info("⚡ Generating partial results report...")
        else:
            logger.info("✅ Analysis complete! Generating final report...")
            
        if args.output:
            reporter.save_json(args.output)
        
        # Only print regular report for non-adaptive modes
        if not args.adaptive:
            reporter.print_report()
        
        # Print summary
        anomaly_count = len([anomaly for anomalies in reporter.anomalies.values() for anomaly in anomalies])
        if anomaly_count > 0:
            logger.info(f"🚨 Found {anomaly_count} anomalies")
        else:
            logger.info("✅ No anomalies detected")
            
        if shutdown_requested:
            logger.info("🔄 Analysis was interrupted but partial results have been saved")

    except KeyboardInterrupt:
        # This shouldn't normally happen due to signal handler, but just in case
        logger.info("\n⚡ Interrupted by user, generating partial results...")
        if reporter and args.output:
            reporter.save_json(args.output)
        if reporter and not args.adaptive:
            reporter.print_report()
    except Exception as e:
        logger.error(f"Error processing log file: {str(e)}")
        raise

if __name__ == '__main__':
    main() 