
import logging
import re
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document

from insurellm.rag.ingestion import IngestionReport


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def normalize_cell(value: str | None) -> str:
    """Normalize whitespace without changing cell meaning."""
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_header(value: str | None) -> str:
    """Normalize a table header for use as a field name."""
    return normalize_cell(value).lower()


def load_tabular_pdf(
    directory: Path,
    report: IngestionReport,
    *,
    doc_type: str = "tabular_record",
    language: str = "en",
) -> list[Document]:
    """Extract PDF tables into one Document per data row."""
    documents = []
    files = sorted(directory.glob("*.pdf"))
    report.files_discovered += len(files)

    for file_path in files:
        file_documents = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()

                    if not tables:
                        report.add_warning(
                            f"No table extracted from "
                            f"{file_path.name}, page {page_number}"
                        )
                        continue

                    for table_index, table in enumerate(tables, start=1):
                        if not table or len(table) < 2:
                            report.add_warning(
                                f"Empty or header-only table in "
                                f"{file_path.name}, page {page_number}, "
                                f"table {table_index}"
                            )
                            continue

                        headers = [
                            normalize_header(cell) for cell in table[0]
                        ]

                        if (
                            any(not header for header in headers)
                            or len(headers) != len(set(headers))
                        ):
                            report.add_warning(
                                f"Invalid or duplicate headers in "
                                f"{file_path.name}, page {page_number}, "
                                f"table {table_index}; table skipped"
                            )
                            continue

                        for row_number, row in enumerate(
                            table[1:], start=1
                        ):
                            if len(row) != len(headers):
                                report.add_warning(
                                    f"Malformed row {row_number} in "
                                    f"{file_path.name}, page {page_number}, "
                                    f"table {table_index}; row skipped"
                                )
                                continue

                            values = [
                                normalize_cell(cell) for cell in row
                            ]

                            if not any(values):
                                report.add_warning(
                                    f"Empty row in {file_path.name}, "
                                    f"page {page_number}; row skipped"
                                )
                                continue

                            record = dict(zip(headers, values))

                            content = "\n".join(
                                f"{header}: {value}"
                                for header, value in record.items()
                                if value
                            )

                            if not content:
                                continue

                            file_documents.append(
                                Document(
                                    page_content=content,
                                    metadata={
                                        "source": file_path.relative_to(
                                            PROJECT_ROOT
                                        ).as_posix(),
                                        "doc_type": doc_type,
                                        "language": language,
                                        "page": page_number,
                                        "record_id": (
                                            f"{file_path.stem}:"
                                            f"page_{page_number}:"
                                            f"table_{table_index}:"
                                            f"row_{row_number}"
                                        ),
                                    },
                                )
                            )

        except Exception as exc:
            report.add_failure(file_path, exc)
            # Do not keep partial output from a failed file.
            continue

        report.files_processed += 1
        documents.extend(file_documents)

    return documents
