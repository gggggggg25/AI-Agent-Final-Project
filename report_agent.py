"""Agent 2: Report Agent (Gemini). Sees only results.json, so it can't invent data."""
import json

from google import genai
from google.genai import types

from config import MODEL_NAME, OUT

SYSTEM = """You are a Report Agent writing for school administrators.
Rules:
- Use ONLY numbers present in the supplied results JSON. Never invent statistics.
- If something is not in the data, say it is unknown.
- Distinguish correlation from causation.
- Do not recommend targeting students by sex, address, or family background.
- Explain that G1/G2 are earlier-period grades, so the model using them is accurate
  but not very actionable, while the model without them shows other drivers.
Output Markdown with exactly these sections, in this order:
# Executive Summary   (max 120 words)
# Methodology
# Findings
# Limitations
# Recommendations    (3-5 actionable items, each tied to a specific finding)
Reference charts with ![caption](filename.png) using the chart filenames in the JSON."""


def write_report(results):
    client = genai.Client()
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents="Results from the Data Scientist Agent:\n" + json.dumps(results, indent=2),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.2,
            max_output_tokens=4000,
        ),
    )
    report = resp.text
    (OUT / "report.md").write_text(report, encoding="utf-8")
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    return report
