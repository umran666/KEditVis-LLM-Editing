"""Generate a revised abstract document from a source .docx.

Preserves student names, registration numbers, batch numbers, guide details,
margins, fonts and alignments, replacing only the abstract body paragraph so the
stated claims match the empirical benchmark findings. The source document is
left strictly untouched.

The .docx files are deliberately not tracked in this repository, so both paths
are required arguments rather than hard-coded absolutes.

Usage:
    python export_doc.py --source path/to/original.docx --output path/to/revised.docx
"""

import argparse
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

# The original abstract always begins with this sentence, which is how the
# paragraph is located in the source document.
ABSTRACT_PREFIX = "Large Language Models frequently encode obsolete"

REVISED_ABSTRACT_TEXT = (
    "Large Language Models frequently encode obsolete or inaccurate factual associations "
    "within their parameters. While locate-then-edit methods such as ROME and MEMIT provide "
    "computationally efficient alternatives to full model retraining, conventional pipelines "
    "rely on static, model-wide layer presets. These fixed presets ignore fact-specific "
    "activation patterns, frequently causing incomplete edits, localized hallucination, or "
    "catastrophic parameter drift. Grounded in recent visual analytics research for model editing, "
    "this capstone project develops an interactive prototype for human-in-the-loop layer "
    "selection. The system extracts internal model signals, specifically layer-wise residual "
    "variance and vocabulary probability distributions, presenting them through coordinated visual "
    "interfaces. Users can evaluate candidate layer ranges across editing success, paraphrase "
    "generalization, and neighborhood locality metrics. The framework incorporates a reversible "
    "model state mechanism and dimensionality reduction to monitor hidden state drift. "
    "Experiments conducted on open-source Transformer architectures using standardized editing "
    "benchmarks demonstrate that interactive telemetry-guided layer selection successfully identifies "
    "viable editing bands to prevent catastrophic failure modes, while multi-context optimization "
    "enhances paraphrase generalization under bounded parameter drift."
)


def build_revised_document(source: Path, output: Path) -> Path:
    """Copy `source` to `output`, replacing only the abstract body paragraph."""
    if not source.exists():
        raise FileNotFoundError(
            f"Source document not found: {source}\n"
            "The .docx sources are not tracked in this repository; pass --source "
            "pointing at the original file."
        )

    document = docx.Document(str(source))
    paragraph = next(
        (p for p in document.paragraphs if p.text.strip().startswith(ABSTRACT_PREFIX)),
        None,
    )
    if paragraph is None:
        raise ValueError(
            f"Could not locate the abstract body paragraph in {source} "
            f"(expected one starting with {ABSTRACT_PREFIX!r})."
        )

    # Capture the original styling before replacing the text, because assigning
    # paragraph.text collapses the run list down to a single new run.
    alignment = paragraph.alignment or WD_ALIGN_PARAGRAPH.JUSTIFY
    runs = paragraph.runs
    font_name = runs[0].font.name if runs and runs[0].font.name else "Times New Roman"
    font_size = runs[0].font.size if runs and runs[0].font.size else Pt(12)

    paragraph.text = REVISED_ABSTRACT_TEXT
    paragraph.alignment = alignment
    for run in paragraph.runs:
        run.font.name = font_name
        run.font.size = font_size

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path, help="original .docx to read")
    parser.add_argument("--output", required=True, type=Path, help="revised .docx to write")
    args = parser.parse_args()

    written = build_revised_document(args.source, args.output)
    print(f"Successfully generated revised abstract document at:\n{written}")


if __name__ == "__main__":
    main()
