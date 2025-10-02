from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# ---------- Value Objects ----------


@dataclass(frozen=True)
class Section:
    title: str | None
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    section_title: str | None
    char_len: int


# ---------- Heuristiken für Struktur ----------

_HEADING_PATTERNS = [
    re.compile(r"^\s*#{1,6}\s+(.+)$"),  # Markdown #, ##, ...
    # 1. / 1.1. / 1.1.1 (optional abschließender Punkt nach Zahlensequenz)
    re.compile(r"^\s*(\d+(?:\.\d+){0,3}\.?)\s+(.+)$"),
    # LAUTSCHRIFT/SEKTIONEN (heurist.)
    re.compile(r"^([A-ZÄÖÜ][A-ZÄÖÜ0-9 \-/]{3,})\s*$"),
]

_SENT_END = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])"
)  # naive, aber robust genug ohne externe Libs


def _is_heading(line: str) -> str | None:
    for pat in _HEADING_PATTERNS:
        m = pat.match(line.strip())
        if m:
            # Bevorzuge die 2. Gruppe (Titel), falls vorhanden; sonst 1.
            if m.lastindex and m.lastindex >= 2 and m.group(2):
                return m.group(2).strip()
            return m.group(1).strip()
    return None


def split_into_sections(text: str) -> list[Section]:
    """Trenne Text in Sektionen anhand typischer Überschriften-Muster."""
    sections: list[Section] = []
    curr_title: str | None = None
    buf: list[str] = []

    lines = text.splitlines()
    for line in lines:
        title = _is_heading(line)
        if title is not None:
            if buf:
                sections.append(Section(curr_title, "\n".join(buf).strip()))
                buf = []
            curr_title = title
        else:
            buf.append(line)
    if buf:
        sections.append(Section(curr_title, "\n".join(buf).strip()))
    if not sections:  # Fallback: gesamter Text als eine Sektion
        sections = [Section(None, text.strip())]
    return sections


def split_into_paragraphs(text: str) -> list[str]:
    """Absatz-Chunking: leere Zeilen trennen Absätze."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras if paras else ([text.strip()] if text.strip() else [])


def split_into_sentences(paragraph: str) -> list[str]:
    """Sehr einfacher Satz-Split (ohne externe NLP-Libs)."""
    paragraph = " ".join(paragraph.split())  # normalisieren
    if not paragraph:
        return []
    parts = re.split(_SENT_END, paragraph)
    # Rückbau: Satzzeichen behalten, aber Split ist schon danach; hier nicht nötig.
    return [s.strip() for s in parts if s.strip()]


# ---------- Chunk-Packer ----------


@dataclass(frozen=True)
class ChunkingParams:
    target_chars: int = 1000
    overlap_chars: int = 150
    max_overhang: int = 200
    merge_threshold: int = 500
    inject_section_titles: bool = True


def _inject_section_title(text: str, section: str | None) -> str:
    if not section:
        return text
    # Titel nur injizieren, wenn noch nicht vorhanden:
    first = text[:200].lower()
    if section.lower() in first:
        return text
    return f"{section}\n\n{text}"


def pack_sentences_to_chunks(
    sentences: Sequence[str],
    section_title: str | None,
    p: ChunkingParams,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    curr: list[str] = []
    curr_len = 0

    for s in sentences:
        s_len = len(s) + (1 if curr else 0)  # +1 für Leerzeichen/Zeilenumbruch
        # Wenn Hinzufügen grob in Ziel passt → rein
        if curr_len + s_len <= p.target_chars:
            curr.append(s)
            curr_len += s_len
            continue
        # Wenn leicht über Ziel, aber innerhalb max_overhang → auch rein
        if curr_len > 0 and (curr_len + s_len - p.target_chars) <= p.max_overhang:
            curr.append(s)
            curr_len += s_len
            continue
        # Sonst: Chunk schließen und Overlap vorbereiten
        if curr:
            chunk_text = " ".join(curr)
            if p.inject_section_titles:
                chunk_text = _inject_section_title(chunk_text, section_title)
            chunks.append(
                Chunk(text=chunk_text, section_title=section_title, char_len=len(chunk_text))
            )

            # Overlap: die letzten overlap_chars Zeichen als Start des nächsten Chunks
            if p.overlap_chars > 0:
                tail = chunk_text[-p.overlap_chars :]
                curr = [tail]
                curr_len = len(tail)
            else:
                curr = []
                curr_len = 0

        # Jetzt aktuellen Satz hinzufügen (kann größer als target sein →
        # alleiniger Chunk beim nächsten Lauf)
        if s:
            if curr_len + len(s) + (1 if curr else 0) <= (p.target_chars + p.max_overhang):
                if curr:
                    curr.append(s)
                    curr_len += len(s) + 1
                else:
                    curr = [s]
                    curr_len = len(s)

    # Rest schließen
    if curr:
        chunk_text = " ".join(curr)
        if p.inject_section_titles:
            chunk_text = _inject_section_title(chunk_text, section_title)
        chunks.append(Chunk(text=chunk_text, section_title=section_title, char_len=len(chunk_text)))

    return chunks


def merge_tiny_neighbors(chunks: list[Chunk], p: ChunkingParams) -> list[Chunk]:
    """Fasse benachbarte Minichunks derselben Sektion zusammen (z. B. Titel + erster Absatz)."""
    if not chunks:
        return []
    merged: list[Chunk] = []
    buf = chunks[0]
    for nxt in chunks[1:]:
        if buf.char_len < p.merge_threshold and nxt.section_title == buf.section_title:
            # zusammenführen
            combined_text = (buf.text + "\n\n" + nxt.text).strip()
            buf = Chunk(
                text=combined_text, section_title=buf.section_title, char_len=len(combined_text)
            )
        else:
            merged.append(buf)
            buf = nxt
    merged.append(buf)
    return merged


def chunk_text_semantic(text: str, params: ChunkingParams | None = None) -> list[Chunk]:
    """Pipeline: Sektion → Absatz → Satz → Packen → Merge tiny.

    Behalte zusätzlich einen Überlappungs‑Tail über Sektionsgrenzen hinweg.
    """
    p = params or ChunkingParams()
    result: list[Chunk] = []
    prev_tail: str | None = None
    for sec in split_into_sections(text):
        paragraphs = split_into_paragraphs(sec.text)
        sentences: list[str] = []
        for para in paragraphs:
            sentences.extend(split_into_sentences(para))
        sec_chunks = pack_sentences_to_chunks(sentences, sec.title, p)

        # Überlappung über Sektionsgrenzen beibehalten
        if prev_tail and sec_chunks:
            first = sec_chunks[0]
            new_text = first.text
            if p.inject_section_titles and sec.title:
                prefix = f"{sec.title}\n\n"
                if new_text.startswith(prefix):
                    new_text = prefix + prev_tail + " " + new_text[len(prefix) :]
                else:
                    new_text = prev_tail + " " + new_text
            else:
                new_text = prev_tail + " " + new_text
            replaced = Chunk(
                text=new_text, section_title=first.section_title, char_len=len(new_text)
            )
            sec_chunks = [replaced] + sec_chunks[1:]

        result.extend(merge_tiny_neighbors(sec_chunks, p))

        # Tail für nächste Sektion aktualisieren
        if result and p.overlap_chars > 0:
            prev_tail = result[-1].text[-p.overlap_chars :]
        else:
            prev_tail = None
    return result


# Eigenschaften:
#
# - Kein I/O, keine Globals, keine externen NLP‑Libs.
# - Überlappung über Zeichen‑Tail (stabil für Cosine‑Retrieval).
# - Section‑Title‑Injection am Chunk‑Anfang (verbessert Kontext/Attribution).
# - Merge tiny (verhindert isolierte Kleinst‑Chunks).
# - Erweiterbar (z. B. „Titelseiten‑Erkennung“ über Heuristik in split_into_sections).
