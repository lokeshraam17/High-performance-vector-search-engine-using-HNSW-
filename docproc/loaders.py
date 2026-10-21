"""
Document File Loaders.

Provides flexible, fault-tolerant ingestion functions for:
- CSV files (with automatic header detection, metadata extraction, and error handling)
- Plain text files (.txt)
- Directories of plain text files
- JSON / JSONL files
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Sequence

from docproc.document import Document

logger = logging.getLogger(__name__)


def load_from_csv(
    filepath: str,
    id_col: str | None = None,
    text_col: str | None = None,
    metadata_cols: Sequence[str] | None = None,
    on_error: str = "raise",
    encoding: str = "utf-8",
) -> list[Document]:
    """
    Load documents from a CSV file.

    Features:
    - Smart column detection: Automatically detects standard ID column candidates
      ('id', 'doc_id', 'document_id', 'ID') and text column candidates
      ('text', 'body', 'content', 'description', 'sentence', 'abstract').
    - Error policy:
      - 'raise': Raises ValueError on malformed rows, missing text, or duplicate IDs.
      - 'skip': Silently skips malformed rows and logs warning.
    - All non-ID and non-text columns are automatically preserved in the document's metadata dict.

    Args:
        filepath: Path to CSV file.
        id_col: Specific name of ID column. If None, detected automatically.
        text_col: Specific name of text column. If None, detected automatically.
        metadata_cols: Optional subset of columns to treat as metadata. If None, all extra columns included.
        on_error: Error policy ('raise' or 'skip'). Defaults to 'raise'.
        encoding: File encoding (default 'utf-8').

    Returns:
        List of loaded Document instances.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If file is empty, missing required columns, or contains malformed data.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    if on_error not in ("raise", "skip"):
        raise ValueError(f"Unknown on_error policy '{on_error}'. Must be 'raise' or 'skip'.")

    documents: list[Document] = []
    seen_ids: set[str] = set()

    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file '{filepath}' is empty or has no header.")

        fieldnames = [col.strip() for col in reader.fieldnames if col]
        fieldnames_lower = {col.lower(): col for col in fieldnames}

        # 1. Resolve ID column
        resolved_id_col = None
        if id_col is not None:
            if id_col in fieldnames:
                resolved_id_col = id_col
            elif id_col.lower() in fieldnames_lower:
                resolved_id_col = fieldnames_lower[id_col.lower()]
            else:
                raise ValueError(f"Specified id_col '{id_col}' not found in CSV headers: {fieldnames}")
        else:
            # Auto-detection candidates
            for candidate in ("id", "doc_id", "document_id", "item_id", "uuid"):
                if candidate in fieldnames_lower:
                    resolved_id_col = fieldnames_lower[candidate]
                    break

        # 2. Resolve Text column
        resolved_text_col = None
        if text_col is not None:
            if text_col in fieldnames:
                resolved_text_col = text_col
            elif text_col.lower() in fieldnames_lower:
                resolved_text_col = fieldnames_lower[text_col.lower()]
            else:
                raise ValueError(f"Specified text_col '{text_col}' not found in CSV headers: {fieldnames}")
        else:
            # Auto-detection candidates
            for candidate in ("text", "body", "content", "description", "document", "sentence", "abstract"):
                if candidate in fieldnames_lower:
                    resolved_text_col = fieldnames_lower[candidate]
                    break

        if resolved_text_col is None:
            raise ValueError(
                f"Could not identify text column in CSV headers: {fieldnames}. "
                f"Please specify text_col explicitly."
            )

        # 3. Parse rows
        for row_idx, row in enumerate(reader, start=2):
            # Check for empty or malformed row
            if not row:
                continue

            # Extract doc_id
            if resolved_id_col is not None:
                raw_id = row.get(resolved_id_col)
                doc_id = str(raw_id).strip() if raw_id is not None else ""
            else:
                # Fallback: synthesize row ID
                doc_id = f"row_{row_idx - 1:04d}"

            if not doc_id:
                msg = f"Row {row_idx}: Document ID in column '{resolved_id_col}' is missing or empty."
                if on_error == "raise":
                    raise ValueError(msg)
                logger.warning(msg)
                continue

            if doc_id in seen_ids:
                msg = f"Row {row_idx}: Duplicate document ID '{doc_id}' encountered."
                if on_error == "raise":
                    raise ValueError(msg)
                logger.warning(msg)
                continue

            # Extract text
            raw_text = row.get(resolved_text_col)
            text = str(raw_text).strip() if raw_text is not None else ""
            if not text:
                msg = f"Row {row_idx}: Missing or empty text in column '{resolved_text_col}' for ID '{doc_id}'."
                if on_error == "raise":
                    raise ValueError(msg)
                logger.warning(msg)
                continue

            # Extract metadata
            metadata: dict[str, Any] = {}
            if metadata_cols is not None:
                for mc in metadata_cols:
                    if mc in row and row[mc] is not None:
                        metadata[mc] = row[mc]
            else:
                # Include all non-ID and non-text columns
                for k, v in row.items():
                    if k and k not in (resolved_id_col, resolved_text_col) and v is not None:
                        metadata[k] = v

            try:
                doc = Document(doc_id=doc_id, text=text, metadata=metadata)
                documents.append(doc)
                seen_ids.add(doc_id)
            except Exception as e:
                if on_error == "raise":
                    raise ValueError(f"Row {row_idx}: Invalid document: {e}") from e
                logger.warning("Row %d: Skipped due to error: %s", row_idx, e)

    return documents


def load_from_text_file(
    filepath: str,
    doc_id: str | None = None,
    encoding: str = "utf-8",
) -> Document:
    """
    Load a single plain text file as a Document.

    Args:
        filepath: Path to the text file.
        doc_id: Optional document ID. Defaults to the filename without extension.
        encoding: File encoding (default 'utf-8').

    Returns:
        Document instance.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Text file not found: {filepath}")

    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Text file '{filepath}' is empty.")

    if doc_id is None:
        doc_id = os.path.splitext(os.path.basename(filepath))[0]

    metadata = {
        "source_file": os.path.abspath(filepath),
        "file_size_bytes": os.path.getsize(filepath),
    }

    return Document(doc_id=doc_id, text=text, metadata=metadata)


def load_from_directory(
    dirpath: str,
    extension: str = ".txt",
    encoding: str = "utf-8",
) -> list[Document]:
    """
    Load all text files matching an extension from a directory.

    Args:
        dirpath: Path to directory.
        extension: File extension filter (e.g. '.txt').
        encoding: File encoding (default 'utf-8').

    Returns:
        List of Document instances.
    """
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f"Directory not found: {dirpath}")

    documents: list[Document] = []
    ext_lower = extension.lower()

    for filename in sorted(os.listdir(dirpath)):
        if filename.lower().endswith(ext_lower):
            path = os.path.join(dirpath, filename)
            if os.path.isfile(path):
                try:
                    doc = load_from_text_file(path, encoding=encoding)
                    documents.append(doc)
                except Exception as e:
                    logger.warning("Skipping file '%s' due to error: %s", filename, e)

    return documents


def load_from_json(
    filepath: str,
    encoding: str = "utf-8",
) -> list[Document]:
    """
    Load documents from a JSON array or JSONL (line-delimited JSON) file.

    Args:
        filepath: Path to JSON or JSONL file.
        encoding: File encoding.

    Returns:
        List of Document instances.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    documents: list[Document] = []

    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # Standard JSON array
            data = json.load(f)
            for item in data:
                documents.append(Document.from_dict(item))
        else:
            # JSONL (line-delimited JSON)
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    item = json.loads(clean_line)
                    documents.append(Document.from_dict(item))
                except Exception as e:
                    raise ValueError(f"Line {line_idx}: Malformed JSON line: {e}") from e

    return documents
