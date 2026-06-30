# Task: Implement Step 1 — Scientific Paper Parser

## Context

I am building a Python pipeline that takes a scientific paper as input and produces a
journal-formatted PDF. The pipeline has 4 stages:

1. **Parse & Extract** ← implement this one only
2. Apply journal rules
3. Render template
4. Generate PDF

---

## Goal for Step 1

Write a Python module `parser.py` that:

- Accepts a file path as input
- Detects whether the file is Markdown (`.md`) or HTML/XML (`.html`, `.xml`)
- Parses it and returns a structured `Paper` object

---

## Input Formats

### Markdown
- Use `mistune` to convert it to an HTML string first
- Enable these plugins: `table`, `footnotes`
- Then parse the resulting HTML with BeautifulSoup4

### HTML / XML
- Parse directly with BeautifulSoup4
- Use `html.parser` for HTML, `xml` for XML

---

## Output: the `Paper` dataclass

Define these dataclasses:

```python
@dataclass
class Author:
    name: str
    affiliation: str | None = None
    email: str | None = None

@dataclass
class Figure:
    caption: str
    src: str | None = None   # image path or url if present

@dataclass
class Section:
    heading: str
    level: int               # 1 = h1, 2 = h2, etc.
    body: str                # plain text content of the section

@dataclass
class Paper:
    title: str
    authors: list[Author]
    abstract: str
    sections: list[Section]
    figures: list[Figure]
    references: list[str]
    raw_html: str            # keep the full HTML for later stages
```

---

## Extraction Logic

Extract from the parsed HTML:

| Field        | Where to look                                                   |
|--------------|-----------------------------------------------------------------|
| `title`      | `<h1>` or `<title>` tag, or a element with class `title`       |
| `authors`    | Elements with class `author`, `authors`, or `contributor`      |
| `abstract`   | `<section>` or `<div>` with id/class `abstract`                |
| `sections`   | All `<h2>`, `<h3>` headings and the paragraphs that follow them |
| `figures`    | All `<figure>` tags; caption from `<figcaption>`               |
| `references` | `<section>` or `<div>` with id/class `references` or `bibliography`, each `<li>` or `<p>` inside it |

Use **fallback logic**: if a semantic class/id is not found, fall back to positional
heuristics (e.g. first `<h1>` = title, last section with many short lines = references).

---

## Entry Point

```python
def parse_paper(filepath: str) -> Paper:
    """
    Main entry point. Detects format, parses, and returns a Paper object.
    Raises ValueError if the format is unsupported.
    """
```

---

## CLI for Testing

Add a `__main__` block so it can be tested from the terminal:

```bash
python parser.py paper.html
python parser.py paper.md
```

It should pretty-print the resulting `Paper` object (use `dataclasses.asdict` +
`json.dumps` with indent=2), truncating `raw_html` to 200 chars in the output.

---

## Requirements

- Python 3.10+
- Libraries: `beautifulsoup4`, `mistune`, `lxml`
- No external API calls
- Keep all logic in a single file `parser.py`
- Add a docstring to every function
- Graceful handling: if a field cannot be extracted, use an empty string / empty list,
  never crash

---

## What NOT to do

- Do not implement stages 2, 3, or 4
- Do not generate any PDF
- Do not apply any journal formatting rules
- Do not modify or rewrite the paper's text content

---

## Deliverable

A single file: `parser.py`
