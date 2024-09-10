import subprocess
import time

# Run main.py in the background
main_process = subprocess.Popen(['python', 'main.py'])

# Check if main.py is running
def is_running(process):
    return process.poll() is None

# Wait a bit to ensure main.py has started
time.sleep(2)

if is_running(main_process):
    print("main.py is running")
else:
    print("main.py failed to start")

# Run manage.py runserver in the foreground and print output to console
subprocess.run(['python', 'manage.py', 'runserver'], stdout=None, stderr=None)

# Optionally, handle cleanup if needed
main_process.terminate()
main_process.wait()