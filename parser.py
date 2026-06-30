"""Scientific paper parser for Markdown, HTML, and XML inputs."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import mistune
from bs4 import BeautifulSoup, NavigableString, Tag


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


_ALL_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_STRUCTURAL_TAGS = {
    "a",
    "b",
    "blockquote",
    "br",
    "caption",
    "code",
    "em",
    "figcaption",
    "figure",
    "hr",
    "i",
    "img",
    "li",
    "ol",
    "p",
    "pre",
    "span",
    "strong",
    "sub",
    "sup",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_META_SECTION_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "bibliographie",
    "bibliography",
    "references",
    "références",
}


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


def _normalize_text(value: str | None) -> str:
    """Normalize text for robust label matching."""

    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    stripped = "".join(character for character in normalized if not unicodedata.combining(character))
    return _clean_text(stripped).casefold().strip(" :;,.\t\n\r-_")


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


def _markdown_to_html(markdown_text: str | None) -> str:
    """Convert Markdown to HTML with table and footnotes support."""

    if not markdown_text:
        return ""
    markdown_parser = mistune.create_markdown(plugins=["table", "footnotes"])
    return markdown_parser(markdown_text)


def _parse_html_content(content: str, fmt: str) -> BeautifulSoup:
    """Create a BeautifulSoup object using parser matching the format."""

    if fmt == "xml":
        try:
            return BeautifulSoup(content, "xml")
        except Exception:
            import warnings

            warnings.warn(
                "XML parsing fell back to html.parser because an XML parser is unavailable. "
                "Install lxml for proper XML/JATS support.",
                UserWarning,
                stacklevel=2,
            )
            return BeautifulSoup(content, "html.parser")
    return BeautifulSoup(content, "html.parser")


def _find_by_semantic_marker(soup: BeautifulSoup, markers: set[str]) -> Tag | None:
    """Find the first element whose id or class matches any marker."""

    normalized_markers = {_normalize_text(marker) for marker in markers}

    def _name_matches(name: str) -> bool:
        normalized_name = _normalize_text(name)
        if normalized_name in normalized_markers:
            return True
        parts = re.split(r"[-_\s]+", normalized_name)
        return any(part in normalized_markers for part in parts if part)

    def _matches(tag: Tag) -> bool:
        if not isinstance(tag, Tag) or tag.name in _STRUCTURAL_TAGS:
            return False

        tag_id = _clean_text(tag.get("id", ""))
        if tag_id and _name_matches(tag_id):
            return True

        classes = tag.get("class") or []
        for class_name in classes:
            if _name_matches(_clean_text(class_name)):
                return True
        return False

    try:
        return soup.find(_matches)
    except Exception:
        return None


def _find_text_heading(soup: BeautifulSoup, markers: set[str]) -> Tag | None:
    """Find a heading-like tag whose visible text matches a marker."""

    normalized_markers = {_normalize_text(marker) for marker in markers}
    for tag in soup.find_all([*list(_ALL_HEADING_TAGS), "p", "div"]):
        text = _normalize_text(tag.get_text(" ", strip=True))
        if text in normalized_markers:
            return tag
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
        text = _clean_text(title_tag.get_text(" ", strip=True))
        if text:
            return text

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


def _tag_text_without_metadata(element: Tag) -> str:
    """Return tag text after removing affiliation/email subtrees."""

    clone = BeautifulSoup(str(element), "html.parser")
    for node in clone.find_all("a", href=lambda href: href and href.startswith("mailto:")):
        node.decompose()
    for node in clone.find_all(class_=lambda classes: classes and any(_normalize_text(value) in {"affiliation", "email"} for value in classes)):
        node.decompose()
    text_parts = [str(node) for node in clone.descendants if isinstance(node, NavigableString)]
    return _clean_text(" ".join(text_parts))


def _extract_authors(soup: BeautifulSoup) -> list[Author]:
    """Extract authors from semantic classes with best-effort parsing."""

    authors: list[Author] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    markers = {"author", "authors", "contributor"}

    elements: list[Tag] = []
    for marker in markers:
        elements.extend(soup.find_all(class_=lambda classes, marker=marker: classes and marker in {_normalize_text(value) for value in classes}))

    for element in elements:
        affiliation_tag = element.find(class_=lambda classes: classes and "affiliation" in {_normalize_text(value) for value in classes})
        email_tag = element.find(class_=lambda classes: classes and "email" in {_normalize_text(value) for value in classes})
        mailto_tag = element.find("a", href=lambda href: href and href.startswith("mailto:"))

        affiliation = _clean_text(affiliation_tag.get_text(" ", strip=True)) if affiliation_tag else None
        email = None
        if email_tag:
            email = _clean_text(email_tag.get_text(" ", strip=True))
        elif mailto_tag and isinstance(mailto_tag, Tag):
            email = _clean_text((mailto_tag.get("href") or "").replace("mailto:", ""))

        text = _tag_text_without_metadata(element)
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


def _extract_container_text(container: Tag, stop_labels: set[str] | None = None) -> str:
    """Collect text from a container, skipping an initial heading if present."""

    stop_labels = stop_labels or set()
    body_parts: list[str] = []
    heading_skipped = False

    for child in container.children:
        if isinstance(child, NavigableString):
            text = _clean_text(str(child))
            if text:
                body_parts.append(text)
            continue

        if not isinstance(child, Tag):
            continue

        if child.name in _ALL_HEADING_TAGS:
            heading_text = _normalize_text(child.get_text(" ", strip=True))
            if not heading_skipped and heading_text in stop_labels:
                heading_skipped = True
                continue
            if heading_text in _META_SECTION_HEADINGS:
                break

        if child.name in {"p", "div", "li", "blockquote"}:
            text = _clean_text(child.get_text(" ", strip=True))
            if text:
                body_parts.append(text)

    return _clean_text(" ".join(body_parts))


def _extract_abstract(soup: BeautifulSoup) -> str:
    """Extract abstract text from semantic container or heading-based fallback."""

    abstract_container = _find_by_semantic_marker(soup, {"abstract"})
    if abstract_container:
        text = _extract_container_text(abstract_container, {"abstract"})
        if text:
            return text

    for heading in soup.find_all(list(_ALL_HEADING_TAGS)):
        heading_text = _normalize_text(heading.get_text(" ", strip=True))
        if heading_text != "abstract":
            continue
        text_parts: list[str] = []
        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag) and sibling.name in _ALL_HEADING_TAGS:
                break
            if isinstance(sibling, Tag) and sibling.name in {"p", "div", "li", "blockquote"}:
                text = _clean_text(sibling.get_text(" ", strip=True))
                if text:
                    text_parts.append(text)
        if text_parts:
            return _clean_text(" ".join(text_parts))

    return ""


def _collect_text_until_next_heading(heading: Tag) -> str:
    """Collect plain text blocks following a heading until the next heading."""

    body_parts: list[str] = []
    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag) and sibling.name in _ALL_HEADING_TAGS:
            break
        if isinstance(sibling, Tag) and sibling.name in {"p", "li", "blockquote", "div"}:
            text = _clean_text(sibling.get_text(" ", strip=True))
            if text:
                body_parts.append(text)
    return _clean_text("\n".join(body_parts))


def _extract_sections(soup: BeautifulSoup) -> list[Section]:
    """Extract sections from h2-h4 headings, skipping meta headings."""

    sections: list[Section] = []

    for heading in soup.find_all(["h2", "h3", "h4"]):
        heading_text = _clean_text(heading.get_text(" ", strip=True))
        if not heading_text:
            continue
        if _normalize_text(heading_text) in _META_SECTION_HEADINGS:
            continue
        try:
            level = int(heading.name[1])
        except Exception:
            level = 2
        body = _collect_text_until_next_heading(heading)
        sections.append(Section(heading=heading_text, level=level, body=body))

    if sections:
        return sections

    paras = soup.find_all("p")
    if not paras:
        return sections

    current_body: list[str] = []
    for para in paras:
        text = _clean_text(para.get_text(" ", strip=True))
        if not text or text.startswith("["):
            continue
        if _normalize_text(text) in _META_SECTION_HEADINGS:
            continue

        current_body.append(text)

        if len("\n".join(current_body)) > 500:
            body_text = _clean_text("\n".join(current_body))
            if body_text:
                heading = f"Section {len(sections) + 1}"
                sections.append(Section(heading=heading, level=2, body=body_text))
            current_body = []

    if current_body:
        body_text = _clean_text("\n".join(current_body))
        if body_text:
            heading = f"Section {len(sections) + 1}"
            sections.append(Section(heading=heading, level=2, body=body_text))

    return sections


def _extract_figures(soup: BeautifulSoup) -> list[Figure]:
    """Extract figures from figure tags with caption and source."""

    figures: list[Figure] = []
    seen_srcs: set[str | None] = set()

    for figure in soup.find_all("figure"):
        figcaption = figure.find("figcaption")
        image = figure.find("img")
        caption = _clean_text(figcaption.get_text(" ", strip=True)) if figcaption else ""
        src = image.get("src") if image and isinstance(image, Tag) else None
        seen_srcs.add(src)
        figures.append(Figure(caption=caption, src=src))

    for img in soup.find_all("img"):
        if not isinstance(img, Tag):
            continue
        if img.find_parent("figure"):
            continue
        src = img.get("src")
        if src in seen_srcs:
            continue
        seen_srcs.add(src)
        caption = ""
        next_p = img.find_next_sibling("p")
        if next_p:
            candidate = _clean_text(next_p.get_text(" ", strip=True))
            if candidate.lower().startswith(("fig", "figure", "image")) or len(candidate) <= 200:
                caption = candidate
        figures.append(Figure(caption=caption, src=src))

    return figures


def _looks_like_reference(text: str) -> bool:
    """Heuristic: does this text look like a bibliographic reference?"""

    has_bracket_num = bool(re.match(r"^\s*\[\d+\]", text))
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", text))
    has_doi = "doi" in text.lower() or "http" in text.lower()
    return has_bracket_num or has_year or has_doi


def _collect_reference_entries_after_heading(heading: Tag) -> list[str]:
    """Collect bibliography entries that appear after a reference heading."""

    entries: list[str] = []
    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag) and sibling.name in _ALL_HEADING_TAGS:
            break
        if isinstance(sibling, Tag) and sibling.name in {"p", "div", "li"}:
            text = _clean_text(sibling.get_text(" ", strip=True))
            if text:
                entries.append(text)
        elif isinstance(sibling, Tag) and sibling.name in {"ol", "ul"}:
            items = [_clean_text(li.get_text(" ", strip=True)) for li in sibling.find_all("li")]
            entries.extend([item for item in items if item])
    return entries


def _fallback_references_from_tail(soup: BeautifulSoup) -> list[str]:
    """Fallback reference extraction from trailing lists or short paragraphs."""

    candidates: list[str] = []

    list_containers = soup.find_all(["ol", "ul"])
    for container in reversed(list_containers):
        items = [_clean_text(li.get_text(" ", strip=True)) for li in container.find_all("li")]
        items = [item for item in items if item]
        if len(items) >= 3 and sum(_looks_like_reference(item) for item in items) >= max(1, len(items) // 2):
            return items

    paragraphs = [_clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    tail = [text for text in paragraphs[-20:] if text]
    reference_tail = [text for text in tail if _looks_like_reference(text)]
    if len(reference_tail) >= 3:
        candidates.extend(reference_tail)

    return candidates


def _extract_references(soup: BeautifulSoup) -> list[str]:
    """Extract references from semantic containers with fallback heuristics."""

    markers = {"references", "bibliography", "références", "bibliographie"}

    references_container = _find_by_semantic_marker(soup, markers)
    if references_container:
        refs = [
            _clean_text(node.get_text(" ", strip=True))
            for node in references_container.find_all(["li", "p"])
        ]
        refs = [ref for ref in refs if ref]
        if refs:
            return refs

    heading = _find_text_heading(soup, markers)
    if heading:
        refs = [ref for ref in _collect_reference_entries_after_heading(heading) if ref]
        if refs:
            return refs

    return _fallback_references_from_tail(soup)


def _is_jats_xml(soup: BeautifulSoup) -> bool:
    """Detect JATS-style XML documents."""

    return bool(
        soup.find("article-meta")
        or soup.find("contrib-group")
        or soup.find("ref-list")
        or soup.find("article")
    )


def _jats_title(soup: BeautifulSoup) -> str:
    """Extract a title from JATS XML."""

    title_tag = soup.find("article-title")
    if title_tag:
        text = _clean_text(title_tag.get_text(" ", strip=True))
        if text:
            return text

    title_tag = soup.find("title")
    if title_tag:
        text = _clean_text(title_tag.get_text(" ", strip=True))
        if text:
            return text

    return ""


def _jats_authors(soup: BeautifulSoup) -> list[Author]:
    """Extract authors from JATS contrib groups."""

    authors: list[Author] = []
    contribs = soup.find_all("contrib")

    for contrib in contribs:
        contrib_type = _clean_text(contrib.get("contrib-type", "")).lower()
        if contrib_type and contrib_type != "author":
            continue

        name_tag = contrib.find("name")
        if name_tag:
            surname = _clean_text(name_tag.find("surname").get_text(" ", strip=True)) if name_tag.find("surname") else ""
            given_names = _clean_text(name_tag.find("given-names").get_text(" ", strip=True)) if name_tag.find("given-names") else ""
            name = _clean_text(" ".join(part for part in [given_names, surname] if part))
        else:
            name = _clean_text(contrib.get_text(" ", strip=True))

        aff_tag = contrib.find("aff")
        email_tag = contrib.find("email")
        affiliation = _clean_text(aff_tag.get_text(" ", strip=True)) if aff_tag else None
        email = _clean_text(email_tag.get_text(" ", strip=True)) if email_tag else None

        if name:
            authors.append(Author(name=name, affiliation=affiliation or None, email=email or None))

    return authors


def _jats_abstract(soup: BeautifulSoup) -> str:
    """Extract the abstract from JATS XML."""

    abstract_tag = soup.find("abstract")
    if not abstract_tag:
        return ""
    return _clean_text(abstract_tag.get_text(" ", strip=True))


def _jats_figures(soup: BeautifulSoup) -> list[Figure]:
    """Extract figures from JATS XML."""

    figures: list[Figure] = []
    for figure in soup.find_all("fig"):
        caption_tag = figure.find("caption")
        graphic = figure.find("graphic") or figure.find("media") or figure.find("img")
        caption = _clean_text(caption_tag.get_text(" ", strip=True)) if caption_tag else ""
        src = None
        if graphic and isinstance(graphic, Tag):
            for attribute in ("href", "xlink:href", "src"):
                value = graphic.get(attribute)
                if value:
                    src = value
                    break
        figures.append(Figure(caption=caption, src=src))
    return figures


def _jats_sections_from_sec(sec: Tag, level: int = 2) -> list[Section]:
    """Recursively extract JATS sections."""

    sections: list[Section] = []
    title_tag = sec.find("title", recursive=False)
    heading = _clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""

    body_parts: list[str] = []
    for child in sec.children:
        if isinstance(child, NavigableString):
            text = _clean_text(str(child))
            if text:
                body_parts.append(text)
            continue
        if not isinstance(child, Tag):
            continue
        if child.name == "title":
            continue
        if child.name == "sec":
            sections.extend(_jats_sections_from_sec(child, min(level + 1, 6)))
            continue
        if child.name in {"p", "list", "boxed-text", "disp-quote", "def-list", "table-wrap"}:
            text = _clean_text(child.get_text(" ", strip=True))
            if text:
                body_parts.append(text)

    body = _clean_text("\n".join(body_parts))
    if heading and _normalize_text(heading) not in _META_SECTION_HEADINGS:
        sections.insert(0, Section(heading=heading, level=level, body=body))

    return sections


def _jats_sections(soup: BeautifulSoup) -> list[Section]:
    """Extract JATS sections from article-body or article."""

    container = soup.find("article-body") or soup.find("body") or soup.find("article") or soup
    root_sections = container.find_all("sec", recursive=False) if isinstance(container, Tag) else []
    if not root_sections and container is soup:
        root_sections = soup.find_all("sec", recursive=False)

    sections: list[Section] = []
    for sec in root_sections:
        sections.extend(_jats_sections_from_sec(sec))
    return sections


def _jats_references(soup: BeautifulSoup) -> list[str]:
    """Extract references from JATS XML."""

    ref_list = soup.find("ref-list")
    if not ref_list:
        return []

    references: list[str] = []
    for ref in ref_list.find_all("ref", recursive=False):
        citation = ref.find(["mixed-citation", "element-citation", "citation"])
        text = _clean_text(citation.get_text(" ", strip=True)) if citation else _clean_text(ref.get_text(" ", strip=True))
        if text:
            references.append(text)
    return references


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


def _build_jats_paper(soup: BeautifulSoup, raw_html: str) -> Paper:
    """Assemble a Paper object from JATS XML."""

    return Paper(
        title=_jats_title(soup),
        authors=_jats_authors(soup),
        abstract=_jats_abstract(soup),
        sections=_jats_sections(soup),
        figures=_jats_figures(soup),
        references=_jats_references(soup),
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

    if fmt == "xml":
        soup = _parse_html_content(content, fmt)
        if _is_jats_xml(soup):
            return _build_jats_paper(soup, content)
        return _build_paper(soup, content)

    if fmt == "html":
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
