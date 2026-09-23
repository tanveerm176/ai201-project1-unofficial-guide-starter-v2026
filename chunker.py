"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

from dataclasses import dataclass
import re

import config
from ingest import Document


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split documents into section-aware, approximately fixed-size chunks.

    Markdown sections beginning with `##` stay together when they fit within
    `config.CHUNK_SIZE`. Longer sections are split using the fallback strategy:
    sentence-aware character windows with up to `config.CHUNK_OVERLAP`
    characters of overlap. This keeps each section heading with its content
    instead of combining unrelated sections in one chunk.

    Chunks are labeled as produced by this function so `app.py chunks` and the
    README can identify the strategy used.
    """
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    sentence_end = re.compile(r"[.!?](?=\s|$)")
    section_heading = re.compile(r"(?m)^## .*$")

    for doc in documents:
        index = 0
        headings = list(section_heading.finditer(doc.text))
        section_ranges = []
        if headings and headings[0].start() > 0:
            section_ranges.append((0, headings[0].start()))
        section_ranges.extend(
            (heading.start(), headings[i + 1].start() if i + 1 < len(headings) else len(doc.text))
            for i, heading in enumerate(headings)
        )
        if not section_ranges:
            section_ranges.append((0, len(doc.text)))

        for section_start, section_end in section_ranges:
            section = doc.text[section_start:section_end].strip()
            start = 0
            while start < len(section):
                target_end = min(start + chunk_size, len(section))
                if target_end == len(section):
                    end = target_end
                else:
                    before_target = list(sentence_end.finditer(section, start, target_end))
                    if before_target:
                        end = before_target[-1].end()
                    else:
                        after_target = sentence_end.search(section, target_end)
                        end = after_target.end() if after_target else target_end

                piece = section[start:end].strip()
                if piece:
                    chunks.append(
                        Chunk(
                            text=piece,
                            source=doc.source,
                            index=index,
                            produced_by="chunker.py::split_documents",
                        )
                    )
                    index += 1

                if end >= len(section):
                    break
                start = max(end - overlap, start + 1)

    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
