from joblib import load
import pandas as pd
import os
import json
import sys
import logging
from typing import List, Dict, Any, Union
from datetime import datetime
import argparse

try:
    # Local feature definitions
    from app.features.columns_nsl_kdd import COLUMNS as NSL_COLUMNS, CATEGORICAL as NSL_CATEGORICAL
except ImportError:
    # Fallback if package import not available
    NSL_COLUMNS = []
    NSL_CATEGORICAL = ['protocol_type', 'service', 'flag']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of NSL-KDD features (41 features) - exclude label and difficulty
NSL_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]


def extract_nsl_features(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only NSL-KDD features from logged traffic.
    Removes metadata like IPs, ports, timestamps.
    """
    features = {}
    
    for feature in NSL_FEATURES:
        if feature in log_entry:
            features[feature] = log_entry[feature]
        else:
            # Set defaults for missing features
            if feature in NSL_CATEGORICAL:
                features[feature] = 'other'
            else:
                features[feature] = 0
    
    return features


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a sample dictionary.
    """
    normalized = extract_nsl_features(sample)
    
    # Ensure categorical features have valid values
    for cat in NSL_CATEGORICAL:
        if cat in normalized and normalized[cat] is None:
            normalized[cat] = "other"
    
    # Convert numeric features to proper types
    for key, value in normalized.items():
        if key not in NSL_CATEGORICAL:
            try:
                normalized[key] = float(value)
            except (ValueError, TypeError):
                normalized[key] = 0.0
    
    return normalized


def predict(model: Union[str, object], sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict on a single sample dictionary.
    """
    # Extract metadata from input
    metadata = {
        'timestamp': sample_dict.get('timestamp', datetime.now().isoformat()),
        'src_ip': sample_dict.get('src_ip', ''),
        'dst_ip': sample_dict.get('dst_ip', ''),
        'src_port': sample_dict.get('src_port', 0),
        'dst_port': sample_dict.get('dst_port', 0),
        'original_pred': sample_dict.get('pred', ''),
        'original_score': sample_dict.get('score_attack', 0)
    }
    
    # Validate and extract NSL features
    nsl_features = validate_sample(sample_dict)
    
    # Load model if path is provided
    if isinstance(model, (str, bytes, os.PathLike)):
        try:
            clf = load(model)
        except Exception as e:
            logger.error("Failed to load model from %s: %s", model, e)
            raise
    else:
        # Assume it's already a loaded estimator
        clf = model
    
    # Create DataFrame from NSL features
    df = pd.DataFrame([nsl_features])
    
    # Make prediction
    try:
        prediction = int(clf.predict(df)[0])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise
    
    # Prepare result with metadata
    result = {
        "prediction": prediction,
        "is_attack": prediction == 1,
        "timestamp": metadata['timestamp'],
        "src_ip": metadata['src_ip'],
        "dst_ip": metadata['dst_ip'],
        "src_port": metadata['src_port'],
        "dst_port": metadata['dst_port'],
        "protocol": nsl_features.get('protocol_type', ''),
        "service": nsl_features.get('service', ''),
        "original_pred": metadata['original_pred'],
        "original_score": metadata['original_score']
    }
    
    # Get probability if available
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(df)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                result["score_attack"] = float(proba[0, 1])
            else:
                result["score_attack"] = 0.5
        except Exception as e:
            logger.warning("predict_proba failed: %s", e)
            result["score_attack"] = 0.5
    else:
        result["score_attack"] = 0.5
    
    return result


def batch_predict(model: Union[str, object], samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict on multiple samples.
    """
    results = []
    for i, sample in enumerate(samples):
        try:
            result = predict(model, sample)
            results.append(result)
            
            # Log attacks in real-time
            if result['is_attack']:
                logger.warning(
                    "🚨 Attack detected: %s:%s -> %s:%s (confidence: %.2f%%)",
                    result['src_ip'], result['src_port'],
                    result['dst_ip'], result['dst_port'],
                    result.get('score_attack', 0) * 100
                )
                
        except Exception as e:
            logger.error("Failed to predict sample %d: %s", i, e)
            results.append({
                "error": str(e),
                "timestamp": sample.get('timestamp', datetime.now().isoformat()),
                "src_ip": sample.get('src_ip', ''),
                "dst_ip": sample.get('dst_ip', '')
            })
    return results


def load_logfile(logfile_path: str) -> List[Dict[str, Any]]:
    """
    Load log entries from a file.
    Supports:
    - JSON array file: [ {...}, {...} ]
    - NDJSON file: {}\n{}\n
    - JSON lines: same as NDJSON
    """
    samples = []
    
    try:
        with open(logfile_path, 'r') as f:
            content = f.read().strip()
            
            if not content:
                logger.error("Logfile is empty: %s", logfile_path)
                return []
            
            # Try to parse as JSON array
            if content.startswith('['):
                data = json.loads(content)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
            else:
                # Parse as NDJSON (newline-delimited JSON)
                samples = []
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            samples.append(entry)
                        except json.JSONDecodeError as e:
                            logger.error("Invalid JSON at line %d: %s", line_num, e)
                            continue
        
        logger.info("Loaded %d log entries from %s", len(samples), logfile_path)
        return samples
        
    except Exception as e:
        logger.error("Failed to load logfile %s: %s", logfile_path, e)
        return []


def detect_attacks_from_logfile(model_path: str, logfile_path: str, 
                               output_file: str = None, 
                               verbose: bool = False) -> Dict[str, Any]:
    """
    Detect attacks from a logfile and return statistics.
    """
    logger.info("🔍 Starting attack detection from: %s", logfile_path)
    logger.info("📦 Using model: %s", model_path)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error("Model file not found: %s", model_path)
        return {"error": "Model file not found"}
    
    # Check if logfile exists
    if not os.path.exists(logfile_path):
        logger.error("Logfile not found: %s", logfile_path)
        return {"error": "Logfile not found"}
    
    # Load log entries
    samples = load_logfile(logfile_path)
    if not samples:
        logger.error("No valid log entries found in %s", logfile_path)
        return {"error": "No valid log entries"}
    
    # Load model
    try:
        model = load(model_path)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return {"error": f"Model load failed: {e}"}
    
    # Make predictions
    results = batch_predict(model, samples)
    
    # Calculate statistics
    total_samples = len(results)
    successful = sum(1 for r in results if 'error' not in r)
    attacks = sum(1 for r in results if r.get('is_attack', False))
    
    # Prepare summary
    summary = {
        "logfile": logfile_path,
        "model": model_path,
        "total_entries": total_samples,
        "successfully_processed": successful,
        "attacks_detected": attacks,
        "attack_percentage": (attacks / successful * 100) if successful > 0 else 0,
        "normal_traffic": successful - attacks,
        "processing_time": datetime.now().isoformat(),
        "detailed_results": results if verbose else []
    }
    
    # Save results if output file specified
    if output_file:
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    "summary": summary,
                    "detections": results
                }, f, indent=2)
            
            logger.info("💾 Results saved to: %s", output_file)
            summary["output_file"] = output_file
        except Exception as e:
            logger.error("Failed to save results: %s", e)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ATTACK DETECTION SUMMARY")
    print("="*60)
    print(f"Logfile:         {logfile_path}")
    print(f"Model:           {model_path}")
    print(f"Total entries:   {total_samples}")
    print(f"Processed:       {successful}")
    print(f"Attacks detected: {attacks} ({attacks/successful*100:.1f}%)")
    print(f"Normal traffic:  {successful - attacks}")
    
    if attacks > 0:
        print(f"\n🚨 DETECTED ATTACKS:")
        for i, result in enumerate(results):
            if result.get('is_attack'):
                print(f"  {i+1}. {result['timestamp']}")
                print(f"     From: {result['src_ip']}:{result['src_port']}")
                print(f"     To:   {result['dst_ip']}:{result['dst_port']}")
                print(f"     Service: {result['protocol']}/{result['service']}")
                print(f"     Confidence: {result.get('score_attack', 0):.1%}")
                if result.get('original_pred'):
                    print(f"     Original system: {result['original_pred']}")
                print()
    
    print("="*60)
    
    return summary


def realtime_detection(model_path: str):
    """
    Real-time attack detection from stdin.
    """
    logger.info("🔍 Starting real-time attack detection")
    logger.info("📦 Using model: %s", model_path)
    
    try:
        model = load(model_path)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return
    
    print("\n🔄 Ready for real-time detection. Enter log entries (one per line):")
    print("   Press Ctrl+C to exit")
    print("-" * 50)
    
    try:
        line_num = 0
        attack_count = 0
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            line_num += 1
            
            try:
                log_entry = json.loads(line)
                result = predict(model, log_entry)
                
                # Display result
                if result['is_attack']:
                    attack_count += 1
                    print(f"\n🚨 ATTACK DETECTED #{attack_count}")
                    print(f"   Entry: {line_num}")
                    print(f"   Time:  {result['timestamp']}")
                    print(f"   From:  {result['src_ip']}:{result['src_port']}")
                    print(f"   To:    {result['dst_ip']}:{result['dst_port']}")
                    print(f"   Type:  {result['protocol']}/{result['service']}")
                    print(f"   Confidence: {result.get('score_attack', 0):.1%}")
                    if result.get('original_pred'):
                        print(f"   Original: {result['original_pred']}")
                    print("-" * 50)
                else:
                    if line_num % 10 == 0:  # Show progress every 10 entries
                        print(f"✓ Processed {line_num} entries, {attack_count} attacks")
                
            except json.JSONDecodeError:
                logger.warning("Invalid JSON at line %d", line_num)
            except Exception as e:
                logger.error("Error at line %d: %s", line_num, e)
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Detection stopped")
        print(f"   Total entries: {line_num}")
        print(f"   Total attacks: {attack_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Intrusion Detection System - Attack Detection from Logfiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --logfile traffic.json --output results.json
  %(prog)s --logfile attacks.ndjson --model models/best_svm.joblib --verbose
  cat live_traffic.json | %(prog)s --realtime
  %(prog)s --test (quick test with built-in sample)
        """
    )
    
    parser.add_argument(
        "--logfile", "-l",
        help="Path to logfile (JSON array or NDJSON format)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="models/best_dt.joblib",
        help="Path to trained model (default: models/best_dt.joblib)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for detection results (JSON format)"
    )
    
    parser.add_argument(
        "--realtime", "-r",
        action="store_true",
        help="Real-time mode (read from stdin)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with all results"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test with a built-in sample log entry"
    )
    
    args = parser.parse_args()
    
    if args.realtime:
        realtime_detection(args.model)
    
    elif args.logfile:
        detect_attacks_from_logfile(
            model_path=args.model,
            logfile_path=args.logfile,
            output_file=args.output,
            verbose=args.verbose
        )
    
    elif args.test:
        # Test with a sample
        sample = {
            "duration": 0.0,
            "protocol_type": "tcp",
            "service": "ftp",
            "flag": "SF",
            "src_bytes": 100,
            "dst_bytes": 2000,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "is_guest_login": 0,
            "count": 2,
            "srv_count": 2,
            "serror_rate": 0,
            "srv_serror_rate": 0,
            "rerror_rate": 0,
            "srv_rerror_rate": 0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 150,
            "dst_host_srv_count": 25,
            "dst_host_same_srv_rate": 0.17,
            "dst_host_diff_srv_rate": 0.03,
            "dst_host_same_src_port_rate": 0.17,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "src_port": 12345,
            "dst_port": 21,
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        print("🧪 Testing with sample log entry...\n")
        result = predict(args.model, sample)
        
        print("📊 Sample Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()