from bu_superagent.domain.services.chunking import (
    ChunkingParams,
    chunk_text_semantic,
    merge_tiny_neighbors,
    pack_sentences_to_chunks,
    split_into_paragraphs,
    split_into_sections,
    split_into_sentences,
)


def test_chunking_respects_target_and_overlap():
    text = (
        "# Titel\n\nAbsatz eins. Satz zwei! Satz drei?\n\n## Untertitel\nNoch ein Absatz."
        " Und noch einer."
    )
    p = ChunkingParams(
        target_chars=50,
        overlap_chars=10,
        max_overhang=10,
        merge_threshold=20,
        inject_section_titles=True,
    )
    chunks = chunk_text_semantic(text, p)

    assert len(chunks) >= 1
    # Überlappung: Folgender Chunk soll Anker aus dem Ende des vorherigen enthalten
    if len(chunks) > 1:
        tail = chunks[0].text[-p.overlap_chars :]
        assert tail in chunks[1].text

    # Titelinjektion:
    assert "Titel" in chunks[0].text


def test_section_detection_markdown_and_numbered():
    txt = """# Einleitung
Dies ist Intro.

1. Überblick
Weiterer Text."""
    secs = split_into_sections(txt)
    assert len(secs) == 2
    assert secs[0].title == "Einleitung"
    assert "Dies ist Intro" in secs[0].text
    assert secs[1].title == "Überblick"


def test_paragraph_and_sentence_split():
    para_text = "Erster Satz. Zweiter Satz! Dritter Satz?"
    paras = split_into_paragraphs(para_text)
    assert paras == [para_text]
    sents = split_into_sentences(para_text)
    assert sents == ["Erster Satz.", "Zweiter Satz!", "Dritter Satz?"]


def test_pack_and_merge_chunks():
    sents = ["kurz"] * 50  # viele kurze Sätze
    p = ChunkingParams(target_chars=20, overlap_chars=5, merge_threshold=30)
    chunks = pack_sentences_to_chunks(sents, section_title="Abschnitt A", p=p)
    assert len(chunks) >= 2
    # Overlap sollte vorhanden sein (der Tail des ersten Chunks kommt im zweiten vor)
    if len(chunks) >= 2:
        tail = chunks[0].text[-5:]
        assert tail in chunks[1].text

    merged = merge_tiny_neighbors(chunks, p)
    assert len(merged) <= len(chunks)


def test_sections_fallback_without_headings():
    txt = "Dies ist ein Text ohne Überschrift. Nur Inhalt."
    secs = split_into_sections(txt)
    assert len(secs) == 1
    assert secs[0].title is None
    assert "Inhalt" in secs[0].text


def test_semantic_pipeline_end_to_end():
    txt = """# Titel

Das hier ist ein kurzer Absatz. Er hat zwei Sätze.

Nächster Absatz ist auch kurz."""
    out = chunk_text_semantic(txt, ChunkingParams(target_chars=80, overlap_chars=10))
    assert out
    # Section-Titel sollte injiziert sein
    assert out[0].text.lower().startswith("titel")
