
import csv
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


logger = logging.getLogger(__name__)

# Project root: the directory containing data/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class IngestionReport:
    """Track file-level outcomes during ingestion."""

    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    warnings: int = 0
    documents_created: int = 0
    failures: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings += 1
        logger.warning(message)

    def add_failure(self, file_path: Path, error: Exception) -> None:
        self.files_failed += 1
        message = (
            f"{file_path.relative_to(PROJECT_ROOT).as_posix()}: "
            f"{type(error).__name__}: {error}"
        )
        self.failures.append(message)
        logger.error("Failed to ingest %s", message)

    def log_summary(self) -> None:
        logger.info(
            "Ingestion summary | discovered=%d | processed=%d | "
            "failed=%d | warnings=%d | documents=%d",
            self.files_discovered,
            self.files_processed,
            self.files_failed,
            self.warnings,
            self.documents_created,
        )

        for failure in self.failures:
            logger.error("Failed file: %s", failure)


def clean_text(text: str) -> str:
    """Apply conservative whitespace cleanup without rewriting content."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_path(file_path: Path) -> str:
    """Return a project-relative source path."""
    return file_path.relative_to(PROJECT_ROOT).as_posix()


def load_pdfs(
    directory: Path,
    report: IngestionReport,
) -> list[Document]:
    """Extract each PDF page into a separate Document."""
    documents = []
    files = sorted(directory.glob("*.pdf"))
    report.files_discovered += len(files)

    for file_path in files:
        file_documents = []

        try:
            reader = PdfReader(file_path)

            if reader.is_encrypted:
                raise ValueError(
                    "PDF is encrypted; no password was supplied"
                )

            page_count = len(reader.pages)

            for page_index in range(page_count):
                page_number = page_index + 1

                try:
                    page = reader.pages[page_index]
                    text = clean_text(page.extract_text() or "")
                except Exception as exc:
                    report.add_warning(
                        f"Could not extract page {page_number} from "
                        f"{source_path(file_path)}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                if not text:
                    report.add_warning(
                        f"No extractable text in "
                        f"{source_path(file_path)}, page {page_number}"
                    )
                    continue

                file_documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source_path(file_path),
                            "doc_type": "company",
                            "language": "ar",
                            "page": page_number,
                        },
                    )
                )

        except Exception as exc:
            report.add_failure(file_path, exc)
            continue

        report.files_processed += 1
        documents.extend(file_documents)

    return documents


def load_markdown(
    directory: Path,
    report: IngestionReport,
) -> list[Document]:
    """Load each Markdown contract as one Document."""
    documents = []
    files = sorted(directory.glob("*.md"))
    report.files_discovered += len(files)

    for file_path in files:
        try:
            text = clean_text(file_path.read_text(encoding="utf-8"))

            if not text:
                report.add_warning(
                    f"Empty Markdown file: {source_path(file_path)}"
                )
                report.files_processed += 1
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_path(file_path),
                        "doc_type": "contract",
                        "language": "en",
                    },
                )
            )
            report.files_processed += 1

        except Exception as exc:
            report.add_failure(file_path, exc)
            continue

    return documents


def load_csvs(
    directory: Path,
    report: IngestionReport,
) -> list[Document]:
    """Convert each CSV product row into a readable Document."""
    documents = []
    files = sorted(directory.glob("*.csv"))
    report.files_discovered += len(files)

    required_headers = {"product_name", "product_group"}

    for file_path in files:
        file_documents = []

        try:
            with file_path.open(
                mode="r",
                encoding="utf-8-sig",
                newline="",
            ) as csv_file:
                reader = csv.DictReader(csv_file)

                if not reader.fieldnames:
                    raise ValueError("CSV has no header row")

                # Normalize header case and surrounding whitespace.
                headers = [
                    header.strip().lower() if header else ""
                    for header in reader.fieldnames
                ]

                if any(not header for header in headers):
                    raise ValueError("CSV contains an empty header")

                if len(headers) != len(set(headers)):
                    raise ValueError(
                        "CSV contains duplicate headers after normalization"
                    )

                missing_headers = required_headers - set(headers)

                if missing_headers:
                    raise ValueError(
                        "Missing required CSV headers: "
                        + ", ".join(sorted(missing_headers))
                    )

                reader.fieldnames = headers

                for row_number, row in enumerate(reader, start=2):
                    # Extra values or missing trailing values indicatea structurally malformed row.
                    if None in row or any(
                        value is None for value in row.values()
                    ):
                        report.add_warning(
                            f"Malformed CSV row {row_number} in "
                            f"{source_path(file_path)}; row skipped"
                        )
                        continue

                    fields = [
                        f"{field_name}: {value.strip()}"
                        for field_name, value in row.items()
                        if value and value.strip()
                    ]

                    if not fields:
                        report.add_warning(
                            f"Empty CSV row {row_number} in "
                            f"{source_path(file_path)}; row skipped"
                        )
                        continue

                    product_name = row["product_name"].strip()
                    product_group = row["product_group"].strip()

                    if not product_name:
                        report.add_warning(
                            f"CSV row {row_number} has no product_name in "
                            f"{source_path(file_path)}; row skipped"
                        )
                        continue

                    file_documents.append(
                        Document(
                            page_content="\n".join(fields),
                            metadata={
                                "source": source_path(file_path),
                                "doc_type": "product",
                                "language": "en",
                                "record_id": (
                                    f"{file_path.stem}:row_{row_number}"
                                ),
                                "row_number": row_number,
                                "product": product_name,
                                **(
                                    {"product_group": product_group}
                                    if product_group
                                    else {}
                                ),
                            },
                        )
                    )

        except Exception as exc:
            # Keep this file's documents out if reading fails partway
            # through. Other CSV files will still be processed.
            report.add_failure(file_path, exc)
            continue

        report.files_processed += 1
        documents.extend(file_documents)

    return documents


def load_documents() -> list[Document]:
    """Load supported source formats into a unified collection."""
    required_directories = [
        DATA_DIR / "company",
        DATA_DIR / "contracts",
        DATA_DIR / "products",
    ]

    missing_directories = [
        directory
        for directory in required_directories
        if not directory.is_dir()
    ]

    if missing_directories:
        paths = ", ".join(str(path) for path in missing_directories)
        raise FileNotFoundError(
            f"Required data directories not found: {paths}"
        )

    report = IngestionReport()

    documents = [
        *load_pdfs(DATA_DIR / "company", report),
        *load_markdown(DATA_DIR / "contracts", report),
        *load_csvs(DATA_DIR / "products", report),
    ]

    report.documents_created = len(documents)
    report.log_summary()

    if not documents:
        raise ValueError(
            f"No documents were loaded from {DATA_DIR}. "
            "Check the ingestion warnings and errors."
        )

    if report.files_failed:
        logger.warning(
            "Ingestion completed with %d failed file(s). "
            "Review the errors before building the index.",
            report.files_failed,
        )

    return documents


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        loaded_documents = load_documents()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    print(f"\nTotal Documents: {len(loaded_documents)}")

    for document in loaded_documents:
        print(
            f"\nSource: {document.metadata['source']}"
            f"\nType: {document.metadata['doc_type']}"
            f"\nPage: {document.metadata.get('page', 'N/A')}"
            f"\nCharacters: {len(document.page_content)}"
            f"\nPreview: {document.page_content[:180]!r}"
        )
