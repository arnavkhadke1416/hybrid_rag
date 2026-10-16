# Source documents

Place the PDF and DOCX files to be loaded in this directory. This is the
project's only default document input location; discovery searches nested
folders recursively. Keep filenames descriptive. The project does not
process these files until you explicitly run an ingestion command.

PDF text is extracted per page so page numbers are retained. DOCX paragraphs
and table rows are read in document order; table cells are joined with ` | `.
Temporary files and unsupported formats are skipped with log messages.
Password-protected, corrupt, empty, or textless files are reported as load
failures. Image-only/scanned PDFs are not OCR'd; use text-based PDFs or
separately approved OCR support.

Run these commands from the project root in PowerShell:

```powershell
python scripts\ingest_documents.py
python scripts\rebuild_indexes.py
```

The first command incrementally detects added, changed, removed, and
unchanged files; the second forces every source file to be reprocessed.
Both commands synchronize the dense and keyword indexes. Review the
reported failures and rerun ingestion after correcting a file.

Loaded records, content hashes, and failures are recorded in the configured
ingestion manifest. Normalized chunks, manifests, and indexes are stored
under `data/`, not beside source documents. Keep private documents in this
directory only if you are authorized to process them. The directory contents
are excluded from version control by the project's `.gitignore`; do not
remove that protection to share document samples.
