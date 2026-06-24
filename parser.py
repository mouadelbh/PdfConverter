"""Scientific paper parser for Markdown, HTML, and XML inputs."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mistune
from bs4 import BeautifulSoup, Tag


@dataclass
class Author:
    """Represents a paper author and optional metadata."""

    name: str
    affiliation: str | None = None
    email: str | None = None


@dataclass
class Figure:
    """Represents a figure discovered in the paper."""

    caption: str
    src: str | None = None


@dataclass
class Section:
    """Represents a paper section with heading and plain-text body."""

    heading: str
    level: int
    body: str


@dataclass
class Paper:
    """Structured representation of a parsed scientific paper."""

    title: str
    authors: list[Author]
    abstract: str
    sections: list[Section]
    figures: list[Figure]
    references: list[str]
    raw_html: str


def _read_text_file(filepath: str) -> str:
    """Read text content from disk, trying multiple encodings."""

    encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1", "cp1252"]
    path = Path(filepath)
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Could not decode {filepath} with any supported encoding")


def _clean_text(value: str | None) -> str:
    """Normalize text by collapsing whitespace and trimming ends."""

    if not value:
        return ""
    return " ".join(value.split())


def _detect_format(filepath: str) -> str:
    """Detect supported input format from file extension."""

    suffix = Path(filepath).suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".xml":
        return "xml"
    raise ValueError(f"Unsupported format: {suffix or 'no extension'}")


def _markdown_to_html(markdown_text: str) -> str:
    """Convert Markdown to HTML with table and footnotes support."""

    markdown_parser = mistune.create_markdown(plugins=["table", "footnotes"])
    return markdown_parser(markdown_text)


def _parse_html_content(content: str, fmt: str) -> BeautifulSoup:
    """Create a BeautifulSoup object using parser matching the format."""

    if fmt == "xml":
        return BeautifulSoup(content, "xml")
    return BeautifulSoup(content, "html.parser")


def _find_by_semantic_marker(soup: BeautifulSoup, markers: set[str]) -> Tag | None:
    """Find the first element whose id or class matches any marker."""

    def _matches(tag: Tag) -> bool:
        tag_id = _clean_text(tag.get("id", "")).lower()
        if tag_id in markers:
            return True
        classes = tag.get("class") or []
        for class_name in classes:
            if _clean_text(class_name).lower() in markers:
                return True
        return False

    try:
        return soup.find(_matches)
    except Exception:
        return None


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract paper title using semantic and positional fallbacks."""

    title_class = _find_by_semantic_marker(soup, {"title"})
    if title_class:
        text = _clean_text(title_class.get_text(" ", strip=True))
        if text:
            return text

    h1 = soup.find("h1")
    if h1:
        text = _clean_text(h1.get_text(" ", strip=True))
        if text:
            return text

    title_tag = soup.find("title")
    if title_tag:
        return _clean_text(title_tag.get_text(" ", strip=True))

    # Fallback: use first substantive paragraph (>30 chars, <200 chars) as title
    # Skip paragraphs that look like figure captions
    for para in soup.find_all("p"):
        text = _clean_text(para.get_text(" ", strip=True))
        is_caption = text.lower().startswith(("figure", "fig.", "[")) or ":" in text[:50]
        if 30 <= len(text) <= 200 and not is_caption:
            return text

    return ""


def _split_possible_authors(raw_text: str) -> list[str]:
    """Split author names from a combined text value when needed."""

    cleaned = _clean_text(raw_text)
    if not cleaned:
        return []

    for delimiter in [";", ",", " and "]:
        if delimiter in cleaned:
            pieces = [_clean_text(piece) for piece in cleaned.split(delimiter)]
            return [piece for piece in pieces if piece]

    return [cleaned]


def _extract_authors(soup: BeautifulSoup) -> list[Author]:
    """Extract authors from semantic classes with best-effort parsing."""

    authors: list[Author] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    markers = {"author", "authors", "contributor"}

    elements: list[Tag] = []
    for marker in markers:
        elements.extend(soup.find_all(class_=lambda c: c and marker in [x.lower() for x in c]))

    for element in elements:
        affiliation_tag = element.find(class_=lambda c: c and "affiliation" in [x.lower() for x in c])
        email_tag = element.find(class_=lambda c: c and "email" in [x.lower() for x in c])
        mailto_tag = element.find("a", href=lambda href: href and href.startswith("mailto:"))

        affiliation = _clean_text(affiliation_tag.get_text(" ", strip=True)) if affiliation_tag else None
        email = None
        if email_tag:
            email = _clean_text(email_tag.get_text(" ", strip=True))
        elif mailto_tag and isinstance(mailto_tag, Tag):
            email = _clean_text((mailto_tag.get("href") or "").replace("mailto:", ""))

        text = _clean_text(element.get_text(" ", strip=True))
        names = _split_possible_authors(text)

        for name in names:
            if not name:
                continue
            record = (name, affiliation or None, email or None)
            if record in seen:
                continue
            seen.add(record)
            authors.append(Author(name=name, affiliation=affiliation or None, email=email or None))

    if authors:
        return authors

    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and isinstance(meta_author, Tag):
        content = _clean_text(meta_author.get("content", ""))
        for name in _split_possible_authors(content):
            authors.append(Author(name=name))

    return authors


def _extract_abstract(soup: BeautifulSoup) -> str:
    """Extract abstract text from semantic container or heading-based fallback."""

    abstract_container = _find_by_semantic_marker(soup, {"abstract"})
    if abstract_container:
        return _clean_text(abstract_container.get_text(" ", strip=True))

    for heading in soup.find_all(["h2", "h3"]):
        heading_text = _clean_text(heading.get_text(" ", strip=True)).lower()
        if heading_text != "abstract":
            continue
        text_parts: list[str] = []
        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag) and sibling.name in {"h1", "h2", "h3"}:
                break
            if isinstance(sibling, Tag) and sibling.name in {"p", "div"}:
                text = _clean_text(sibling.get_text(" ", strip=True))
                if text:
                    text_parts.append(text)
        if text_parts:
            return _clean_text(" ".join(text_parts))

    return ""


def _collect_text_until_next_heading(heading: Tag) -> str:
    """Collect plain text blocks following a heading until next heading."""

    body_parts: list[str] = []
    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag) and sibling.name in {"h1", "h2", "h3"}:
            break
        if isinstance(sibling, Tag) and sibling.name in {"p", "li", "blockquote"}:
            text = _clean_text(sibling.get_text(" ", strip=True))
            if text:
                body_parts.append(text)
        if isinstance(sibling, Tag) and sibling.name == "div":
            text = _clean_text(sibling.get_text(" ", strip=True))
            if text:
                body_parts.append(text)
    return _clean_text("\n".join(body_parts))


def _extract_sections(soup: BeautifulSoup) -> list[Section]:
    """Extract h2/h3 sections or fall back to grouping content by heuristic."""

    sections: list[Section] = []

    # Try explicit h2/h3 headings first
    for heading in soup.find_all(["h2", "h3"]):
        heading_text = _clean_text(heading.get_text(" ", strip=True))
        if not heading_text:
            continue
        try:
            level = int(heading.name[1])
        except Exception:
            level = 2
        body = _collect_text_until_next_heading(heading)
        sections.append(Section(heading=heading_text, level=level, body=body))

    if sections:
        return sections

    # Fallback: group paragraphs into synthetic sections by length patterns
    # Treat short consecutive paragraphs as intro, longer blocks as sections
    paras = soup.find_all("p")
    if not paras:
        return sections

    current_body: list[str] = []
    for para in paras:
        text = _clean_text(para.get_text(" ", strip=True))
        if not text or text.startswith("["):
            continue
        current_body.append(text)

        # If we've accumulated enough text (>500 chars), start a new section
        if len("\n".join(current_body)) > 500:
            body_text = _clean_text("\n".join(current_body))
            if body_text:
                # Generate a synthetic section heading based on content
                heading = f"Section {len(sections) + 1}"
                sections.append(Section(heading=heading, level=2, body=body_text))
            current_body = []

    # Capture remaining paragraphs
    if current_body:
        body_text = _clean_text("\n".join(current_body))
        if body_text:
            heading = f"Section {len(sections) + 1}"
            sections.append(Section(heading=heading, level=2, body=body_text))

    return sections


def _extract_figures(soup: BeautifulSoup) -> list[Figure]:
    """Extract figures from figure tags with caption and source."""

    figures: list[Figure] = []
    for figure in soup.find_all("figure"):
        figcaption = figure.find("figcaption")
        image = figure.find("img")
        caption = _clean_text(figcaption.get_text(" ", strip=True)) if figcaption else ""
        src = image.get("src") if image and isinstance(image, Tag) else None
        figures.append(Figure(caption=caption, src=src))
    return figures


def _fallback_references_from_tail(soup: BeautifulSoup) -> list[str]:
    """Fallback reference extraction from trailing lists or short paragraphs."""

    candidates: list[str] = []

    list_containers = soup.find_all(["ol", "ul"])
    for container in reversed(list_containers):
        items = [_clean_text(li.get_text(" ", strip=True)) for li in container.find_all("li")]
        items = [item for item in items if item]
        if len(items) >= 3:
            return items

    paragraphs = [_clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    tail = [text for text in paragraphs[-20:] if text]
    short_tail = [text for text in tail if len(text) <= 160]
    if len(short_tail) >= 3:
        candidates.extend(short_tail)

    return candidates


def _extract_references(soup: BeautifulSoup) -> list[str]:
    """Extract references from semantic containers with fallback heuristics."""

    references_container = _find_by_semantic_marker(soup, {"references", "bibliography"})
    if references_container:
        refs = [
            _clean_text(node.get_text(" ", strip=True))
            for node in references_container.find_all(["li", "p"])
        ]
        refs = [ref for ref in refs if ref]
        if refs:
            return refs

    return _fallback_references_from_tail(soup)


def _build_paper(soup: BeautifulSoup, raw_html: str) -> Paper:
    """Assemble a Paper object from parsed soup and raw HTML."""

    return Paper(
        title=_extract_title(soup),
        authors=_extract_authors(soup),
        abstract=_extract_abstract(soup),
        sections=_extract_sections(soup),
        figures=_extract_figures(soup),
        references=_extract_references(soup),
        raw_html=raw_html,
    )


def parse_paper(filepath: str) -> Paper:
    """Detect format, parse input, and return structured Paper data."""

    fmt = _detect_format(filepath)
    content = _read_text_file(filepath)

    if fmt == "markdown":
        raw_html = _markdown_to_html(content)
        soup = _parse_html_content(raw_html, "html")
        return _build_paper(soup, raw_html)

    if fmt in {"html", "xml"}:
        soup = _parse_html_content(content, fmt)
        return _build_paper(soup, content)

    raise ValueError(f"Unsupported format: {fmt}")


def _paper_to_pretty_json(paper: Paper) -> str:
    """Serialize a Paper object to JSON while truncating raw HTML."""

    payload = asdict(paper)
    raw_html = payload.get("raw_html", "") or ""
    payload["raw_html"] = raw_html[:200] + ("..." if len(raw_html) > 200 else "")
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _main(argv: list[str]) -> int:
    """Run a tiny CLI wrapper for parser testing."""

    if len(argv) != 2:
        print("Usage: python parser.py <paper.md|paper.html|paper.xml>")
        return 1

    filepath = argv[1]
    try:
        paper = parse_paper(filepath)
    except Exception as error:
        print(f"Error: {error}")
        return 1

    print(_paper_to_pretty_json(paper))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
