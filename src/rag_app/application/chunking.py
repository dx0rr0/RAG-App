import re


def _segments_from_transcript(result, max_chars=800, overlap=80):
    segments = result.get("segments") or []
    chunks = []
    current = []
    current_start = None
    current_length = 0

    def flush():
        nonlocal current, current_start, current_length
        text = " ".join(part.strip() for part in current if part.strip()).strip()
        if text:
            chunks.append({"content": text, "timestamp_seconds": current_start})
        current, current_start, current_length = [], None, 0

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = segment.get("start")
        try:
            start = max(0, int(float(start))) if start is not None else None
        except (ValueError, TypeError):
            start = None
        if current and current_length + len(text) + 1 > max_chars:
            previous = " ".join(current)
            previous_start = current_start
            flush()
            if overlap and previous:
                tail = previous[-overlap:]
                current = [tail]
                current_length = len(tail)
                current_start = previous_start
        if not current:
            current_start = start
        if len(text) > max_chars:
            for piece in _text_chunks(text, max_chars, overlap):
                if current:
                    current.append(piece)
                    flush()
                else:
                    current = [piece]
                    current_start = start
                    flush()
            continue
        current.append(text)
        current_length += len(text) + 1
    flush()
    return chunks


def _text_chunks(text, size, overlap):
    text = re.sub(r"\s+", " ", text).strip()
    result = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + size)
        if end < len(text):
            boundary = text.rfind(" ", cursor + size // 2, end)
            if boundary > cursor:
                end = boundary
        result.append(text[cursor:end].strip())
        if end >= len(text):
            break
        cursor = max(cursor + 1, end - overlap)
    return result


def transcript_chunks(result, max_chars=800, overlap=80):
    chunks = _segments_from_transcript(result, max_chars, overlap)
    if chunks:
        return chunks
    return [{"content": part, "timestamp_seconds": None}
            for part in _text_chunks(result.get("text", ""), max_chars, overlap)]
