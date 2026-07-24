import os
import subprocess
import sys
import venv

VENV_DIR = "venv"

def run_command(command, env=None):
    subprocess.check_call(command, shell=False, env=env)

def main():
    print("Setting up project...")

    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print("Virtual environment already exists. Skipping creation.")

    if os.name == "nt":
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(VENV_DIR, "bin", "python")

    print("Upgrading pip...")
    run_command([python_executable, "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing requirements...")
    run_command([python_executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("Setup complete.")

if __name__ == "__main__":
    main()