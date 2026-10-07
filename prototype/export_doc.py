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

# The original abstract always begins with this sentence, which is how the
# paragraph is located in the source document.
ABSTRACT_PREFIX = "Large Language Models frequently encode obsolete"

REVISED_ABSTRACT_TEXT = (
    "Large Language Models frequently encode obsolete or inaccurate factual associations "
    "within their parameters. While locate-then-edit methods such as ROME and MEMIT provide "
    "computationally efficient alternatives to full model retraining, conventional pipelines "
    "rely on static, model-wide layer presets. These fixed presets ignore fact-specific "
    "activation patterns and can produce different editing outcomes across facts. "
    "Grounded in recent visual analytics research for model editing, "
    "this capstone project develops an interactive prototype for human-in-the-loop layer "
    "selection. The system extracts internal model signals, specifically layer-wise residual "
    "variance, MLP cosine similarity, and vocabulary probability distributions, presenting them through coordinated visual "
    "interfaces. Users can evaluate candidate layer ranges across editing success, paraphrase "
    "generalization, and neighborhood locality metrics. The framework incorporates a reversible "
    "model state mechanism and dimensionality reduction to monitor hidden state drift. "
    "A historical ten-fact GPT-2-XL CounterFact pilot compares fixed, telemetry-guided, and "
    "seeded-random layer windows alongside editing profiles. Telemetry-guided selection did not "
    "demonstrate superiority over fixed presets. A profile with matched optimization settings "
    "achieved the same aggregate behavioral scores as the context profiles, leaving the "
    "incremental benefit of context fitting and consistency unresolved. These findings support "
    "an inspectable workflow for comparing editing configurations through behavioral evaluation "
    "and measurements of selected rewrite-tensor changes."
)


def build_revised_document(source: Path, output: Path) -> Path:
    """Copy `source` to `output`, replacing only the abstract body paragraph."""
    if not source.exists():
        raise FileNotFoundError(
            f"Source document not found: {source}\n"
            "The .docx sources are not tracked in this repository; pass --source "
            "pointing at the original file."
        )
    if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
        raise ValueError("The output must be a separate file; the original document is preserved.")

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

    # Keep inherited paragraph formatting and the original text run's style.
    runs = paragraph.runs
    if runs:
        first = next((run for run in runs if run.text), runs[0])
        first.text = REVISED_ABSTRACT_TEXT
        for run in runs:
            if run is not first:
                run.text = ""
    else:
        paragraph.add_run(REVISED_ABSTRACT_TEXT)

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
