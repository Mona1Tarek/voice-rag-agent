
import logging
import sys
from pathlib import Path

from insurellm.rag.ingestion import IngestionReport
from insurellm.rag.tabular_pdf import load_tabular_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMPLOYEES_DIR = PROJECT_ROOT / "data" / "employees"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not EMPLOYEES_DIR.is_dir():
        raise FileNotFoundError(
            f"Test data directory not found: {EMPLOYEES_DIR}"
        )

    report = IngestionReport()

    documents = load_tabular_pdf(
        EMPLOYEES_DIR,
        report,
        doc_type="employee_record",
        language="en",
    )

    report.documents_created = len(documents)
    report.log_summary()

    print(f"\nTotal Documents: {len(documents)}")

    for index, document in enumerate(documents[:100], start=1):
        print(f"\n{'=' * 60}")
        print(f"Document {index}")
        print(f"Source: {document.metadata['source']}")
        print(f"Record ID: {document.metadata['record_id']}")
        print(f"Page: {document.metadata['page']}")
        print(f"Type: {document.metadata['doc_type']}")
        print(f"Content:\n{document.page_content}")

    if report.files_failed:
        raise SystemExit(
            f"\nTest completed with {report.files_failed} failed file(s)."
        )


if __name__ == "__main__":
    main()
