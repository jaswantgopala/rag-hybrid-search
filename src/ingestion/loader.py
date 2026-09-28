import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from pypdf import PdfReader
import markdown as md_lib

@dataclass
class LoadedDoc:
    source_file: str
    file_type: str
    text: str
    section_heading: str | None = None
    page_number: int | None = None

def load_markdown(path: Path) -> list[LoadedDoc]:
    raw = path.read_text(encoding="utf-8")
    html = md_lib.markdown(raw)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return [LoadedDoc(source_file=path.name, file_type="markdown", text=text)]

def load_text(path: Path) -> list[LoadedDoc]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [LoadedDoc(source_file=path.name, file_type="text", text=text)]

def load_html(path: Path) -> list[LoadedDoc]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator="\n")
    return [LoadedDoc(source_file=path.name, file_type="html", text=text)]

def load_pdf(path: Path) -> list[LoadedDoc]:
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(LoadedDoc(
                source_file=path.name,
                file_type="pdf",
                text=text,
                page_number=i + 1
            ))
    return docs

LOADERS = {
    ".md": load_markdown,
    ".txt": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}

def load_all_documents(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> list[LoadedDoc]:
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    all_docs: list[LoadedDoc] = []
    for file_path in raw_path.iterdir():
        ext = file_path.suffix.lower()
        loader = LOADERS.get(ext)
        if not loader:
            print(f"Skipping unsupported file type: {file_path.name}")
            continue
        try:
            docs = loader(file_path)
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} chunk(s) from {file_path.name}")
        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")

    # Persist processed versions so re-indexing doesn't need re-upload
    out_file = processed_path / "processed_docs.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(asdict(doc)) + "\n")

    print(f"\nSaved {len(all_docs)} processed documents to {out_file}")
    return all_docs

if __name__ == "__main__":
    load_all_documents()