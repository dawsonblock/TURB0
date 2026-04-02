import sys
import platform
import json
import os
import importlib.util

def check_apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def check_mlx_version():
    try:
        import mlx
        return getattr(mlx, "__version__", "unknown")
    except ImportError:
        return None

def main():
    # Ensure current directory is in sys.path so we can import turboquant
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
        
    errors = []
    tq_version = None
    try:
        # Instead of 'import turboquant', check if we find it as a source package
        spec = importlib.util.find_spec("turboquant")
        if spec:
            import turboquant
            tq_version = getattr(turboquant, "__version__", "dev")
        else:
            errors.append("Cannot import turboquant")
    except Exception as e:
        errors.append(f"Import failed: {e}")

    results = {
        "apple_silicon": check_apple_silicon(),
        "mlx_version": check_mlx_version(),
        "turboquant_version": tq_version,
        "python_version": sys.version.split('\n')[0],
        "platform": platform.platform(),
        "errors": errors
    }
    
    strict = "--strict" in sys.argv
    if strict:
        if not results["apple_silicon"]:
            print("ERROR: TurboQuant requires Apple Silicon (M1/M2/M3/M4) for production runtime.")
            sys.exit(1)
        if results["mlx_version"] is None:
            print("ERROR: MLX is not installed.")
            sys.exit(1)
            
    if "--json" in sys.argv:
        print(json.dumps(results, indent=2))
    else:
        for k, v in results.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
