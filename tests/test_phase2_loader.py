"""Phase 2 loader coverage for DOCX/Markdown."""

from docx import Document as DocxDocument

from core.ingestion.loader import DocumentLoader


def test_docx_loader(tmp_path):
    path = tmp_path / "sample.docx"
    doc = DocxDocument()
    doc.add_heading("Title", level=1)
    doc.add_paragraph("Body text.")
    doc.save(path)
    docs = DocumentLoader(str(tmp_path)).load_single_document(str(path))
    assert docs and "Body text." in docs[0].page_content
    assert docs[0].metadata["page"] == 0


def test_markdown_loader(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("# Heading\nBody\n", encoding="utf-8")
    docs = DocumentLoader(str(tmp_path)).load_single_document(str(path))
    assert docs and "Body" in docs[0].page_content
