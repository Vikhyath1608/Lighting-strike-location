import schedule
import time
import subprocess
import os

def job():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_directory, 'Main.py')
    subprocess.run(['python', script_path], check=True)

# Schedule the job every 24 hours
schedule.every(24).hours.do(job)

print("Scheduler started.")

while True:
    schedule.run_pending()
    time.sleep(1)
