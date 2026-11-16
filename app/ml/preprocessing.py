import re
import unicodedata


def normalize_description(description: str) -> str:
    if not description:
        return ""

    text = unicodedata.normalize("NFC", description).strip()

    # Remove informações de agência e conta.
    text = re.sub(
        r"\s+Ag[eê]ncia\s*:.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove identificador numérico + nome do titular + CPF/CNPJ.
    text = re.sub(
        r"\s*-\s*[\d\s.]+\s+"
        r"[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’\s.-]*?"
        r"\s*-\s*"
        r"(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"
        r"|\d{3}\.\d{3}\.\d{3}-\d{2})",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove espaços duplicados e separadores sobrando.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = text.strip(" -")

    return text.strip()
