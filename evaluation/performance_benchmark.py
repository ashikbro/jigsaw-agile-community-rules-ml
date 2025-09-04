#!/usr/bin/env python3
"""
Performance Benchmarking Tool for Content Moderation System
Tests model performance, API response times, and system throughput
"""

import time
import requests
import json
import statistics
import concurrent.futures
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from collections import defaultdict
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Individual benchmark test result"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    response_size_bytes: Optional[int] = None
    status_code: Optional[int] = None
    custom_metrics: Optional[Dict[str, Any]] = None

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    start_time: datetime
    end_time: datetime
    total_tests: int
    successful_tests: int
    failed_tests: int
    results: List[BenchmarkResult]
    summary_stats: Dict[str, Any]
    system_metrics: Dict[str, Any]

class APIBenchmark:
    """API performance benchmarking"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_endpoint(self, num_requests: int = 100) -> List[BenchmarkResult]:
        """Benchmark health endpoint"""
        results = []
        
        for i in range(num_requests):
            start_time = datetime.now()
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=10)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() * 1000
                
                result = BenchmarkResult(
                    test_name=f"health_check_{i+1}",
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration,
                    success=response.status_code == 200,
                    status_code=response.status_code,
                    response_size_bytes=len(response.content)
                )
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() * 1000
                
                result = BenchmarkResult(
                    test_name=f"health_check_{i+1}",
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration,
                    success=False,
                    error_message=str(e)
                )
            
            results.append(result)
            
        return results
    
    def test_prediction_endpoint(self, test_texts: List[str], num_iterations: int = 10) -> List[BenchmarkResult]:
        """Benchmark prediction endpoint"""
        results = []
        
        for iteration in range(num_iterations):
            for i, text in enumerate(test_texts):
                start_time = datetime.now()
                try:
                    payload = {"text": text}
                    response = self.session.post(
                        f"{self.base_url}/predict",
                        json=payload,
                        timeout=30
                    )
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    
                    custom_metrics = {}
                    if response.status_code == 200:
                        data = response.json()
                        custom_metrics = {
                            'prediction': data.get('prediction'),
                            'confidence': data.get('confidence'),
                            'text_length': len(text)
                        }
                    
                    result = BenchmarkResult(
                        test_name=f"prediction_{iteration+1}_{i+1}",
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration,
                        success=response.status_code == 200,
                        status_code=response.status_code,
                        response_size_bytes=len(response.content),
                        custom_metrics=custom_metrics
                    )
                    
                except Exception as e:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    
                    result = BenchmarkResult(
                        test_name=f"prediction_{iteration+1}_{i+1}",
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration,
                        success=False,
                        error_message=str(e),
                        custom_metrics={'text_length': len(text)}
                    )
                
                results.append(result)
                
        return results
    
    def test_multilingual_endpoint(self, test_texts: Dict[str, str], num_iterations: int = 5) -> List[BenchmarkResult]:
        """Benchmark multilingual prediction endpoint"""
        results = []
        
        for iteration in range(num_iterations):
            for lang, text in test_texts.items():
                start_time = datetime.now()
                try:
                    payload = {"text": text}
                    response = self.session.post(
                        f"{self.base_url}/predict/multilingual",
                        json=payload,
                        timeout=30
                    )
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    
                    custom_metrics = {'expected_language': lang, 'text_length': len(text)}
                    if response.status_code == 200:
                        data = response.json()
                        custom_metrics.update({
                            'detected_language': data.get('detected_language'),
                            'prediction': data.get('prediction'),
                            'confidence': data.get('confidence'),
                            'translation_used': data.get('translation_used', False)
                        })
                    
                    result = BenchmarkResult(
                        test_name=f"multilingual_{lang}_{iteration+1}",
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration,
                        success=response.status_code == 200,
                        status_code=response.status_code,
                        response_size_bytes=len(response.content),
                        custom_metrics=custom_metrics
                    )
                    
                except Exception as e:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    
                    result = BenchmarkResult(
                        test_name=f"multilingual_{lang}_{iteration+1}",
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration,
                        success=False,
                        error_message=str(e),
                        custom_metrics={'expected_language': lang, 'text_length': len(text)}
                    )
                
                results.append(result)
                
        return results

class LoadTester:
    """Concurrent load testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def run_concurrent_test(self, 
                          endpoint: str, 
                          payload: Dict, 
                          num_threads: int = 10, 
                          requests_per_thread: int = 10) -> List[BenchmarkResult]:
        """Run concurrent load test"""
        results = []
        results_lock = threading.Lock()
        
        def worker(thread_id: int):
            session = requests.Session()
            thread_results = []
            
            for i in range(requests_per_thread):
                start_time = datetime.now()
                try:
                    response = session.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                        timeout=30
                    )
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    
                    result = BenchmarkResult(
                        test_name=f"load_test_t{thread_id}_r{i+1}",
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration,
                        success=response.status_code == 200,
                        status_code=response.status_code,
                        response_size_bytes=len(response.content),
                        custom_metrics={'thread_id': thread_id}
                    )
                    
                except Exception as e:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    
                    result = BenchmarkResult(
                        test_name=f"load_test_t{thread_id}_r{i+1}",
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration,
                        success=False,
                        error_message=str(e),
                        custom_metrics={'thread_id': thread_id}
                    )
                
                thread_results.append(result)
            
            with results_lock:
                results.extend(thread_results)
        
        # Start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        return results

class BenchmarkAnalyzer:
    """Analyzes benchmark results and generates reports"""
    
    @staticmethod
    def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        durations = [r.duration_ms for r in successful_results]
        
        analysis = {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'failed_tests': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100 if results else 0,
            'duration_stats': {},
            'error_analysis': {},
            'throughput': {}
        }
        
        if durations:
            analysis['duration_stats'] = {
                'mean_ms': statistics.mean(durations),
                'median_ms': statistics.median(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'std_dev_ms': statistics.stdev(durations) if len(durations) > 1 else 0,
                'percentile_95_ms': np.percentile(durations, 95),
                'percentile_99_ms': np.percentile(durations, 99)
            }
        
        # Error analysis
        error_counts = defaultdict(int)
        for result in failed_results:
            error_type = result.error_message or f"HTTP_{result.status_code}"
            error_counts[error_type] += 1
        
        analysis['error_analysis'] = dict(error_counts)
        
        # Throughput calculation
        if successful_results:
            total_duration = (max(r.end_time for r in results) - 
                            min(r.start_time for r in results)).total_seconds()
            
            analysis['throughput'] = {
                'requests_per_second': len(successful_results) / total_duration if total_duration > 0 else 0,
                'total_duration_seconds': total_duration
            }
        
        return analysis
    
    @staticmethod
    def generate_report(suite: BenchmarkSuite, output_file: Optional[str] = None) -> str:
        """Generate detailed benchmark report"""
        report = f"""
# Benchmark Report: {suite.suite_name}

## Summary
- **Start Time**: {suite.start_time}
- **End Time**: {suite.end_time}
- **Total Duration**: {suite.end_time - suite.start_time}
- **Total Tests**: {suite.total_tests}
- **Successful Tests**: {suite.successful_tests}
- **Failed Tests**: {suite.failed_tests}
- **Success Rate**: {suite.successful_tests / suite.total_tests * 100:.2f}%

## Performance Statistics
"""
        
        stats = suite.summary_stats.get('duration_stats', {})
        if stats:
            report += f"""
### Response Time Statistics
- **Mean**: {stats.get('mean_ms', 0):.2f} ms
- **Median**: {stats.get('median_ms', 0):.2f} ms
- **Min**: {stats.get('min_ms', 0):.2f} ms
- **Max**: {stats.get('max_ms', 0):.2f} ms
- **95th Percentile**: {stats.get('percentile_95_ms', 0):.2f} ms
- **99th Percentile**: {stats.get('percentile_99_ms', 0):.2f} ms
- **Standard Deviation**: {stats.get('std_dev_ms', 0):.2f} ms
"""
        
        throughput = suite.summary_stats.get('throughput', {})
        if throughput:
            report += f"""
### Throughput
- **Requests per Second**: {throughput.get('requests_per_second', 0):.2f}
- **Total Duration**: {throughput.get('total_duration_seconds', 0):.2f} seconds
"""
        
        errors = suite.summary_stats.get('error_analysis', {})
        if errors:
            report += "\n### Error Analysis\n"
            for error_type, count in errors.items():
                report += f"- **{error_type}**: {count} occurrences\n"
        
        # System metrics
        if suite.system_metrics:
            report += f"""
## System Metrics During Test
- **CPU Usage**: {suite.system_metrics.get('cpu_percent', 0):.1f}%
- **Memory Usage**: {suite.system_metrics.get('memory_percent', 0):.1f}%
- **Memory Used**: {suite.system_metrics.get('memory_used_gb', 0):.2f} GB
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    @staticmethod
    def create_performance_charts(results: List[BenchmarkResult], output_dir: str = "."):
        """Create performance visualization charts"""
        if not results:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Response time distribution
        successful_results = [r for r in results if r.success]
        durations = [r.duration_ms for r in successful_results]
        
        if durations:
            plt.figure(figsize=(12, 8))
            
            # Histogram
            plt.subplot(2, 2, 1)
            plt.hist(durations, bins=30, alpha=0.7, color='skyblue')
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (ms)')
            plt.ylabel('Frequency')
            
            # Box plot
            plt.subplot(2, 2, 2)
            plt.boxplot(durations)
            plt.title('Response Time Box Plot')
            plt.ylabel('Response Time (ms)')
            
            # Time series
            plt.subplot(2, 2, 3)
            timestamps = [r.start_time for r in successful_results]
            plt.plot(timestamps, durations, alpha=0.7)
            plt.title('Response Time Over Time')
            plt.xlabel('Time')
            plt.ylabel('Response Time (ms)')
            plt.xticks(rotation=45)
            
            # Success rate over time
            plt.subplot(2, 2, 4)
            window_size = max(1, len(results) // 20)
            success_rates = []
            windows = []
            
            for i in range(0, len(results), window_size):
                window = results[i:i+window_size]
                success_rate = sum(1 for r in window if r.success) / len(window) * 100
                success_rates.append(success_rate)
                windows.append(window[0].start_time)
            
            plt.plot(windows, success_rates, marker='o')
            plt.title('Success Rate Over Time')
            plt.xlabel('Time')
            plt.ylabel('Success Rate (%)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / 'performance_charts.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance charts saved to {output_path / 'performance_charts.png'}")

class BenchmarkRunner:
    """Main benchmark runner"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_benchmark = APIBenchmark(base_url)
        self.load_tester = LoadTester(base_url)
        self.analyzer = BenchmarkAnalyzer()
    
    def run_full_benchmark_suite(self) -> BenchmarkSuite:
        """Run complete benchmark suite"""
        logger.info("Starting full benchmark suite")
        start_time = datetime.now()
        
        all_results = []
        
        # Test data
        test_texts = [
            "This is a normal comment",
            "You are stupid and ugly",
            "Great job on this project!",
            "I hate this so much, it's terrible",
            "This product is amazing, highly recommended"
        ]
        
        multilingual_texts = {
            'english': "This is a test message",
            'spanish': "Este es un mensaje de prueba",
            'french': "Ceci est un message de test",
            'german': "Dies ist eine Testnachricht"
        }
        
        # System metrics before test
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        # Run tests
        logger.info("Testing health endpoint...")
        all_results.extend(self.api_benchmark.test_health_endpoint(50))
        
        logger.info("Testing prediction endpoint...")
        all_results.extend(self.api_benchmark.test_prediction_endpoint(test_texts, 5))
        
        logger.info("Testing multilingual endpoint...")
        all_results.extend(self.api_benchmark.test_multilingual_endpoint(multilingual_texts, 3))
        
        logger.info("Running load test...")
        load_results = self.load_tester.run_concurrent_test(
            "/predict",
            {"text": "This is a load test message"},
            num_threads=5,
            requests_per_thread=10
        )
        all_results.extend(load_results)
        
        end_time = datetime.now()
        
        # Analyze results
        summary_stats = self.analyzer.analyze_results(all_results)
        
        suite = BenchmarkSuite(
            suite_name="Full API Benchmark Suite",
            start_time=start_time,
            end_time=end_time,
            total_tests=len(all_results),
            successful_tests=len([r for r in all_results if r.success]),
            failed_tests=len([r for r in all_results if not r.success]),
            results=all_results,
            summary_stats=summary_stats,
            system_metrics=system_metrics
        )
        
        logger.info("Benchmark suite completed")
        return suite

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Performance Benchmark Tool")
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--output-dir', default='.', help='Output directory for reports')
    parser.add_argument('--report-format', choices=['markdown', 'json'], default='markdown', help='Report format')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run benchmark
    runner = BenchmarkRunner(args.url)
    suite = runner.run_full_benchmark_suite()
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.report_format == 'markdown':
        report_file = output_path / f"benchmark_report_{timestamp}.md"
        report = runner.analyzer.generate_report(suite, str(report_file))
        print(report)
    
    # Save JSON data
    json_file = output_path / f"benchmark_data_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'suite': {
                **asdict(suite),
                'start_time': suite.start_time.isoformat(),
                'end_time': suite.end_time.isoformat(),
                'results': [
                    {
                        **asdict(result),
                        'start_time': result.start_time.isoformat(),
                        'end_time': result.end_time.isoformat()
                    }
                    for result in suite.results
                ]
            }
        }, f, indent=2)
    
    # Create charts
    runner.analyzer.create_performance_charts(suite.results, args.output_dir)
    
    logger.info(f"Benchmark completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()