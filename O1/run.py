import os
import subprocess
import sys

VENV_DIR = "venv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_TO_RUN = os.path.join(BASE_DIR, "scripts", "main.py")


def get_python_executable():
    """Return correct Python path based on OS"""
    if os.name == "nt":  # Windows
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:  # macOS / Linux
        return os.path.join(VENV_DIR, "bin", "python")


def main():
    print("Running project...")

    python_executable = get_python_executable()

    # Check if venv exists
    if not os.path.exists(python_executable):
        print("❌ Virtual environment not found.")
        print("👉 Please run: python setup.py first")
        sys.exit(1)

    try:
        subprocess.check_call([python_executable, SCRIPT_TO_RUN])
    except subprocess.CalledProcessError as e:
        print("❌ Error while running script:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()