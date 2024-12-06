import psutil
import time

# Get the current process
process = psutil.Process()

while True:
    # Get the process's overall CPU usage as a percentage of total system CPUs
    cpu_usage = process.cpu_percent(interval=1)
    
    # Get per-core CPU usage
    per_core_usage = psutil.cpu_percent(interval=None, percpu=True)
    
    print(f"Process CPU usage: {cpu_usage}%")
    print(f"Per-core CPU usage: {per_core_usage}")
    time.sleep(1)
