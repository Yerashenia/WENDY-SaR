"""
Batch-convert .docx files to .pdf so they can be fed into the KG
extraction pipeline (which only reads PDFs).

Usage:
    python convert_docx_to_pdf.py <docx_folder> <pdf_folder>

Example:
    python convert_docx_to_pdf.py O1/data/docx O1/data/pdfs

Behaviour:
    - On Windows / macOS: uses docx2pdf (requires MS Word installed).
    - On Linux: uses LibreOffice headless (requires `soffice` on PATH).
    - Creates the output folder if it doesn't exist.
    - Skips files that have already been converted (same name, .pdf exists).
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


def convert_windows_or_mac(docx_folder: str, pdf_folder: str):
    """
    Convert using docx2pdf (drives MS Word in the background).
    Requires: pip install docx2pdf
    Requires: Microsoft Word installed.
    """
    try:
        from docx2pdf import convert
    except ImportError:
        print("docx2pdf is not installed. Install it with:")
        print("    pip install docx2pdf")
        sys.exit(1)

    print(f"Converting all .docx files in {docx_folder} -> {pdf_folder}")
    convert(docx_folder, pdf_folder)


def convert_linux(docx_folder: str, pdf_folder: str):
    """
    Convert using LibreOffice headless mode.
    Requires LibreOffice installed (provides the `soffice` command).

    On Ubuntu/Debian: sudo apt install libreoffice
    """
    soffice_path = None
    for candidate in ["soffice", "libreoffice"]:
        check = subprocess.run(
            ["which", candidate],
            capture_output=True,
            text=True,
        )
        if check.returncode == 0 and check.stdout.strip():
            soffice_path = candidate
            break

    if not soffice_path:
        print("LibreOffice (soffice) was not found on PATH.")
        print("Install it first, e.g.: sudo apt install libreoffice")
        sys.exit(1)

    docx_files = sorted(Path(docx_folder).glob("*.docx"))

    if not docx_files:
        print(f"No .docx files found in {docx_folder}")
        return

    for docx_file in docx_files:
        expected_pdf = Path(pdf_folder) / (docx_file.stem + ".pdf")

        if expected_pdf.exists():
            print(f"Skipping (already converted): {docx_file.name}")
            continue

        print(f"Converting: {docx_file.name}")

        result = subprocess.run(
            [
                soffice_path,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(pdf_folder),
                str(docx_file),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  Failed to convert {docx_file.name}")
            print(f"  {result.stderr.strip()}")
        else:
            print(f"  Done: {expected_pdf.name}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_docx_to_pdf.py <docx_folder> <pdf_folder>")
        sys.exit(1)

    docx_folder = sys.argv[1]
    pdf_folder = sys.argv[2]

    if not Path(docx_folder).exists():
        print(f"Input folder not found: {docx_folder}")
        sys.exit(1)

    os.makedirs(pdf_folder, exist_ok=True)

    system_name = platform.system()

    print(f"Detected OS: {system_name}")

    if system_name in ("Windows", "Darwin"):
        convert_windows_or_mac(docx_folder, pdf_folder)
    elif system_name == "Linux":
        convert_linux(docx_folder, pdf_folder)
    else:
        print(f"Unsupported OS: {system_name}")
        sys.exit(1)

    pdf_count = len(list(Path(pdf_folder).glob("*.pdf")))
    print(f"\nConversion complete. {pdf_count} PDF file(s) now in {pdf_folder}")


if __name__ == "__main__":
    main()