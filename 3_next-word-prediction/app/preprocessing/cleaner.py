import re


def clean_line(line: str) -> str:
    line = line.lower()

    line = re.sub(r"[^a-zA-Z0-9\s]", "", line)

    line = re.sub(r"\s+", " ", line)

    return line.strip()


def clean_text(text: str) -> str:
    lines = text.splitlines()

    cleaned_lines = [clean_line(line) for line in lines if line.strip()]

    return "\n".join(cleaned_lines)
