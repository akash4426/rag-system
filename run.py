import subprocess
import time
import sys

def main():
    print("Starting the 23-Component RAG System...")
    
    # Start the FastAPI backend
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"],
    )
    
    # Give the backend a second to start
    time.sleep(2)
    
    # Start the React UI
    ui_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="ui"
    )
    
    try:
        backend_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down RAG System...")
        backend_process.terminate()
        ui_process.terminate()

if __name__ == "__main__":
    main()
