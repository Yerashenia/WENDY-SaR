from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
folder = os.getenv("pdf_folder")

print("Folder:", repr(folder))
print("Exists:", Path(folder).exists() if folder else "N/A")
print("Is dir:", Path(folder).is_dir() if folder else "N/A")

pdfs = list(Path(folder).glob("*.pdf")) if folder else []
print("PDFs found directly in folder:", len(pdfs))
for p in pdfs:
    print(" -", p.name)

# check for nested folders too, just in case
subdirs = [d for d in Path(folder).iterdir() if d.is_dir()] if folder else []
print("Subdirectories:", [d.name for d in subdirs])