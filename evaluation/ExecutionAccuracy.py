import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError, ServiceUnavailable, AuthError
import json

# Configure logging with both file and console output
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging with both file and console handlers.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"query_comparison_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - important messages only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

logger = setup_logging()


class QueryComparator:
    """Compare Neo4j Cypher queries by executing them and comparing results."""
    
    def __init__(self, uri: str, user: str, password: str, timeout: int = 30):
        """
        Initialize the QueryComparator.
        
        Args:
            uri: Neo4j database URI
            user: Neo4j username
            password: Neo4j password
            timeout: Query timeout in seconds
        """
        self.uri = uri
        self.timeout = timeout
        try:
            self.driver = GraphDatabase.driver(
                uri, 
                auth=(user, password),
                connection_timeout=timeout
            )
            # Test connection
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {uri}")
        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {uri}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def run_query(self, query: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            
        Returns:
            Tuple of (result list or None, error message or None)
        """
        if not query or not isinstance(query, str):
            error_msg = "Invalid query: query must be a non-empty string"
            logger.error(error_msg)
            return None, error_msg
            
        query = query.strip()
        if not query:
            error_msg = "Invalid query: query is empty after stripping whitespace"
            logger.error(error_msg)
            return None, error_msg
            
        try:
            with self.driver.session() as session:
                result = session.run(query, timeout=self.timeout)
                records = [record.data() for record in result]
                logger.debug(f"Query executed successfully, returned {len(records)} records")
                return records, None
        except CypherSyntaxError as e:
            error_msg = f"Syntax error: {str(e)}"
            logger.error(f"Syntax error in query:\n{query}\nError: {e}")
            return None, error_msg
        except ServiceUnavailable as e:
            error_msg = f"Database unavailable: {str(e)}"
            logger.error(f"Database unavailable while executing query: {e}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            logger.error(f"Error executing query:\n{query}\nError: {e}")
            return None, error_msg

    def normalize_result(self, result: List[Dict[str, Any]]) -> List[str]:
        """
        Normalize query results for comparison.
        
        Args:
            result: List of result records
            
        Returns:
            Sorted list of stringified values
        """
        if not result:
            return []
            
        normalized = []
        for record in result:
            if not isinstance(record, dict):
                logger.warning(f"Unexpected record type: {type(record)}")
                continue
            # Extract and stringify all values
            for value in record.values():
                normalized.append(str(value))
        
        # Sort for consistent comparison
        return sorted(normalized)

    def compare_queries(
        self, 
        query1: str, 
        query2: str,
        index: str,
        logic_type: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compare two Cypher queries by executing them and comparing results.
        
        Args:
            query1: First query to compare
            query2: Second query to compare
            index: Index of the query being compared
            logic_type: Logic type being tested
            verbose: If True, print detailed comparison info
            
        Returns:
            Dictionary with comparison results and error details
        """
        result_info = {
            'match': False,
            'error': None,
            'error_type': None,
            'query1_error': None,
            'query2_error': None,
            'query1_result_count': 0,
            'query2_result_count': 0
        }
        
        try:
            # Execute both queries
            result1, error1 = self.run_query(query1)
            result2, error2 = self.run_query(query2)

            # Track errors
            if error1:
                result_info['query1_error'] = error1
                result_info['error_type'] = 'query1_execution_error'
                logger.warning(f"[{logic_type}][{index}] Query 1 failed: {error1}")
                
            if error2:
                result_info['query2_error'] = error2
                result_info['error_type'] = 'query2_execution_error'
                logger.warning(f"[{logic_type}][{index}] Query 2 (ground truth) failed: {error2}")

            # Check if either query failed
            if result1 is None or result2 is None:
                result_info['error'] = 'One or both queries failed to execute'
                if verbose:
                    logger.warning(f"[{logic_type}][{index}] One or both queries failed to execute")
                return result_info

            # Record result counts
            result_info['query1_result_count'] = len(result1)
            result_info['query2_result_count'] = len(result2)

            # Normalize results
            result1_normalized = self.normalize_result(result1)
            result2_normalized = self.normalize_result(result2)

            # Compare
            are_equal = result1_normalized == result2_normalized
            result_info['match'] = are_equal
            
            if not are_equal:
                result_info['error_type'] = 'result_mismatch'
                logger.info(
                    f"[{logic_type}][{index}] MISMATCH - "
                    f"Query1: {len(result1)} records, Query2: {len(result2)} records"
                )
            else:
                logger.debug(f"[{logic_type}][{index}] MATCH - Both queries returned identical results")
            
            if verbose:
                logger.info(f"[{logic_type}][{index}] Query 1 returned {len(result1)} records")
                logger.info(f"[{logic_type}][{index}] Query 2 returned {len(result2)} records")
                logger.info(f"[{logic_type}][{index}] Queries are {'identical' if are_equal else 'different'}")

        except Exception as e:
            error_msg = f"Unexpected error during comparison: {str(e)}"
            logger.error(f"[{logic_type}][{index}] {error_msg}")
            result_info['error'] = error_msg
            result_info['error_type'] = 'comparison_error'

        return result_info


def read_json(file_path: str) -> Optional[Dict]:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data or None if failed
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    if not path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return None
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.debug(f"Successfully loaded JSON from {file_path}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def save_results_to_json(results: Dict, output_path: str):
    """
    Save detailed results to a JSON file.
    
    Args:
        results: Results dictionary to save
        output_path: Path to output JSON file
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")


def run_comparison_suite(
    uri: str,
    user: str,
    password: str,
    base_path: str,
    logic_types: Dict[int, str],
    output_dir: str = "results"
) -> Dict[str, Dict]:
    """
    Run comparison suite for all logic types.
    
    Args:
        uri: Neo4j URI
        user: Neo4j username
        password: Neo4j password
        base_path: Base directory path for test files
        logic_types: Dictionary mapping logic type IDs to names
        output_dir: Directory to save result files
        
    Returns:
        Dictionary with comparison results and statistics
    """
    base_path = Path(base_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'summary': {},
        'detailed_results': {}
    }
    
    try:
        with QueryComparator(uri, user, password) as comparator:
            for logic_id, logic_name in logic_types.items():
                logger.info("="*80)
                logger.info(f"Processing logic type {logic_id}: {logic_name}")
                logger.info("="*80)

                # Please update this
                test_file = base_path / f"Gemini3_{logic_name}.ckpt.json"
                ground_truth_file = base_path / f"Ground_truth_{logic_name}.json"
                
                # Load test data
                test_data = read_json(str(test_file))
                ground_truth_data = read_json(str(ground_truth_file))
                
                if test_data is None or ground_truth_data is None:
                    logger.error(f"Failed to load data for logic type {logic_name}")
                    results['summary'][logic_name] = {
                        "status": "error",
                        "message": "Failed to load test files"
                    }
                    continue
                
                # Initialize tracking for this logic type
                total = len(test_data)
                matches = 0
                mismatches = []
                execution_errors = []
                syntax_errors = []
                
                query_details = {}
                
                # Compare queries
                for i in range(total):
                    key = str(i)
                    
                    if key not in test_data or key not in ground_truth_data:
                        logger.warning(f"[{logic_name}][{key}] Missing key in test or ground truth data")
                        execution_errors.append({
                            'index': key,
                            'error': 'Missing key in data files'
                        })
                        continue
                    
                    try:
                        comparison_result = comparator.compare_queries(
                            test_data[key], 
                            ground_truth_data[key],
                            index=key,
                            logic_type=logic_name
                        )
                        
                        # Store detailed results
                        query_details[key] = {
                            'match': comparison_result['match'],
                            'error_type': comparison_result['error_type'],
                            'query1_result_count': comparison_result['query1_result_count'],
                            'query2_result_count': comparison_result['query2_result_count'],
                            'query1_error': comparison_result['query1_error'],
                            'query2_error': comparison_result['query2_error']
                        }
                        
                        if comparison_result['match']:
                            matches += 1
                        else:
                            # Categorize the error
                            if comparison_result['error_type'] == 'result_mismatch':
                                mismatches.append({
                                    'index': key,
                                    'query1_count': comparison_result['query1_result_count'],
                                    'query2_count': comparison_result['query2_result_count']
                                })
                            elif 'execution_error' in comparison_result['error_type']:
                                execution_errors.append({
                                    'index': key,
                                    'error_type': comparison_result['error_type'],
                                    'query1_error': comparison_result['query1_error'],
                                    'query2_error': comparison_result['query2_error']
                                })
                            elif 'syntax' in str(comparison_result.get('query1_error', '')).lower() or \
                                 'syntax' in str(comparison_result.get('query2_error', '')).lower():
                                syntax_errors.append({
                                    'index': key,
                                    'query1_error': comparison_result['query1_error'],
                                    'query2_error': comparison_result['query2_error']
                                })
                            
                    except Exception as e:
                        logger.error(f"[{logic_name}][{key}] Error comparing query: {e}")
                        execution_errors.append({
                            'index': key,
                            'error': str(e)
                        })
                
                # Calculate statistics
                total_errors = len(mismatches) + len(execution_errors) + len(syntax_errors)
                accuracy = matches / total if total > 0 else 0
                
                # Store summary for this logic type
                results['summary'][logic_name] = {
                    "total_queries": total,
                    "matches": matches,
                    "total_errors": total_errors,
                    "mismatches": len(mismatches),
                    "execution_errors": len(execution_errors),
                    "syntax_errors": len(syntax_errors),
                    "accuracy": accuracy,
                    "mismatch_indices": [item['index'] for item in mismatches],
                    "execution_error_indices": [item['index'] for item in execution_errors],
                    "syntax_error_indices": [item['index'] for item in syntax_errors]
                }
                
                # Store detailed results
                results['detailed_results'][logic_name] = {
                    'mismatches': mismatches,
                    'execution_errors': execution_errors,
                    'syntax_errors': syntax_errors,
                    'all_queries': query_details
                }
                
                # Log summary for this logic type
                logger.info(f"\n{'='*80}")
                logger.info(f"Summary for {logic_name}:")
                logger.info(f"  Total queries: {total}")
                logger.info(f"  Matches: {matches} ({accuracy:.2%})")
                logger.info(f"  Total errors: {total_errors}")
                logger.info(f"    - Result mismatches: {len(mismatches)}")
                logger.info(f"    - Execution errors: {len(execution_errors)}")
                logger.info(f"    - Syntax errors: {len(syntax_errors)}")
                
                if mismatches:
                    logger.info(f"  Mismatch indices: {[item['index'] for item in mismatches]}")
                if execution_errors:
                    logger.info(f"  Execution error indices: {[item['index'] for item in execution_errors]}")
                if syntax_errors:
                    logger.info(f"  Syntax error indices: {[item['index'] for item in syntax_errors]}")
                logger.info(f"{'='*80}\n")
    
    except Exception as e:
        logger.error(f"Error in comparison suite: {e}")
        raise
    
    # Save results to JSON
    results_file = output_path / f"comparison_results_{timestamp}.json"
    save_results_to_json(results, str(results_file))
    
    return results


def print_final_summary(results: Dict):
    """
    Print a comprehensive final summary of all results.
    
    Args:
        results: Results dictionary from comparison suite
    """
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    summary = results.get('summary', {})
    
    # Overall statistics
    total_queries = 0
    total_matches = 0
    total_mismatches = 0
    total_execution_errors = 0
    total_syntax_errors = 0
    
    # Print per-logic-type summary
    for logic_name, stats in summary.items():
        if stats.get("status") == "error":
            print(f"\n{logic_name}: ERROR - {stats.get('message', 'Unknown error')}")
        else:
            total_queries += stats["total_queries"]
            total_matches += stats["matches"]
            total_mismatches += stats["mismatches"]
            total_execution_errors += stats["execution_errors"]
            total_syntax_errors += stats["syntax_errors"]
            
            print(f"\n{logic_name}:")
            print(f"  Accuracy: {stats['matches']}/{stats['total_queries']} ({stats['accuracy']:.2%})")
            print(f"  Errors: {stats['total_errors']}")
            print(f"    - Mismatches: {stats['mismatches']}")
            print(f"    - Execution errors: {stats['execution_errors']}")
            print(f"    - Syntax errors: {stats['syntax_errors']}")
            
            if stats['mismatch_indices']:
                print(f"  Mismatch indices: {stats['mismatch_indices']}")
            if stats['execution_error_indices']:
                print(f"  Execution error indices: {stats['execution_error_indices']}")
            if stats['syntax_error_indices']:
                print(f"  Syntax error indices: {stats['syntax_error_indices']}")
    
    # Print overall summary
    if total_queries > 0:
        print("\n" + "="*80)
        print("OVERALL STATISTICS:")
        print(f"  Total queries tested: {total_queries}")
        print(f"  Total matches: {total_matches} ({total_matches/total_queries:.2%})")
        print(f"  Total errors: {total_queries - total_matches}")
        print(f"    - Result mismatches: {total_mismatches}")
        print(f"    - Execution errors: {total_execution_errors}")
        print(f"    - Syntax errors: {total_syntax_errors}")
        print("="*80)
    
    logger.info("Final summary printed to console")


# Main execution
if __name__ == "__main__":
    # Configuration
    NEO4J_URI = ""
    NEO4J_USER = ""
    NEO4J_PASSWORD = ""
    BASE_PATH = r""
    OUTPUT_DIR = "results"
    
    LOGIC_TYPES = {
        1: "0p"
    }
    
    try:
        logger.info("Starting query comparison suite")
        logger.info(f"Neo4j URI: {NEO4J_URI}")
        logger.info(f"Base path: {BASE_PATH}")
        logger.info(f"Testing {len(LOGIC_TYPES)} logic types")
        
        # Run comparison suite
        results = run_comparison_suite(
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD,
            BASE_PATH,
            LOGIC_TYPES,
            OUTPUT_DIR
        )
        
        # Print final summary
        print_final_summary(results)
        
        logger.info("Query comparison suite completed successfully")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise