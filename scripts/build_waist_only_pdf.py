"""Build docs/WAIST_ONLY_CHANGES.pdf with ReportLab."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "WAIST_ONLY_CHANGES.pdf"

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#5B6B86")
GREEN = colors.HexColor("#2F7654")
LIGHT_GREEN = colors.HexColor("#EAF5EF")
PALE = colors.HexColor("#F4F7FB")
LINE = colors.HexColor("#CBD5E1")
RED = colors.HexColor("#B42318")


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title="Waist-Only Underwear Sizing - Change Guide",
            author="Dimensions Check project",
            subject="Complete comparison with the original four-measurement sizing design",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates(PageTemplate(id="guide", frames=[frame], onPage=self._page))

    def _page(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(self.leftMargin, A4[1] - 12 * mm, A4[0] - self.rightMargin, A4[1] - 12 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, A4[1] - 9 * mm, "Dimensions Check - waist_only branch")
        canvas.drawRightString(A4[0] - self.rightMargin, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=28,
    leading=34, alignment=TA_CENTER, textColor=INK, spaceAfter=16,
))
styles.add(ParagraphStyle(
    "CoverSub", parent=styles["Normal"], fontSize=13, leading=19,
    alignment=TA_CENTER, textColor=MUTED,
))
styles.add(ParagraphStyle(
    "H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18,
    leading=23, textColor=INK, spaceBefore=4, spaceAfter=10, keepWithNext=True,
))
styles.add(ParagraphStyle(
    "H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13.5,
    leading=18, textColor=GREEN, spaceBefore=10, spaceAfter=6, keepWithNext=True,
))
styles.add(ParagraphStyle(
    "Bodyx", parent=styles["BodyText"], fontSize=9.6, leading=14.2,
    textColor=INK, spaceAfter=7,
))
styles.add(ParagraphStyle(
    "Smallx", parent=styles["BodyText"], fontSize=8.3, leading=12,
    textColor=MUTED, spaceAfter=5,
))
styles.add(ParagraphStyle(
    "Codex", parent=styles["Code"], fontName="Courier", fontSize=7.5,
    leading=10.5, leftIndent=8, rightIndent=8, borderColor=LINE, borderWidth=.5,
    borderPadding=8, backColor=PALE, spaceBefore=4, spaceAfter=8,
))
styles.add(ParagraphStyle(
    "Callout", parent=styles["BodyText"], fontSize=9.2, leading=13.5,
    leftIndent=9, rightIndent=9, borderColor=GREEN, borderWidth=.8,
    borderPadding=9, backColor=LIGHT_GREEN, textColor=INK, spaceBefore=5, spaceAfter=9,
))


def p(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def bullets(items):
    out = []
    for item in items:
        out.append(Paragraph(f"- {item}", ParagraphStyle(
            "bullet-temp", parent=styles["Bodyx"], leftIndent=12, firstLineIndent=-7,
            spaceAfter=4,
        )))
    return out


def table(rows, widths=None, header=True, font_size=8.2):
    t = Table([[p(str(cell), "Smallx") for cell in row] for row in rows], colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .45, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    for row in range(1 if header else 0, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    t.setStyle(TableStyle(commands))
    return t


story = []
story += [Spacer(1, 52 * mm), p("Waist-Only Underwear Sizing", "CoverTitle")]
story += [p("Complete change guide for the <b>waist_only</b> branch", "CoverSub")]
story += [Spacer(1, 9 * mm)]
story += [p(
    "A precise comparison with the original four-measurement estimator documented in "
    "<b>docs/CODEBASE_GUIDE.pdf</b>. This guide covers implementation, behavior, API impact, "
    "tests, calibration, operating constraints, ambiguity, migration, and rollback.",
    "CoverSub",
)]
story += [Spacer(1, 32 * mm), p("Repository: wackademia/dimensions-check", "CoverSub")]
story += [p("Branch: waist_only", "CoverSub"), p("Generated: 22 September 2026", "CoverSub")]
story += [PageBreak()]

story += [p("Document map", "H1x")]
story += [table([
    ["Section", "Purpose"],
    ["1. Executive summary", "What changed and what did not"],
    ["2. Baseline and scope", "The exact original design used for comparison"],
    ["3. Behavioral contract", "Inputs, validation, matching, and outputs"],
    ["4. Algorithm comparison", "Before and after logic in detail"],
    ["5. Reference ranges", "Tolerance intervals, gaps, and overlaps"],
    ["6. File-by-file changes", "Implementation, UI, tests, and documentation"],
    ["7. Pipeline impact", "What remains unchanged in camera and CV processing"],
    ["8. Verification", "Automated tests and acceptance checks"],
    ["9. Risks and operations", "Limitations of deciding size from waist alone"],
    ["10. Migration and rollback", "How to deploy or reverse the branch"],
    ["Appendix", "Payload examples and decision cases"],
], widths=[48 * mm, 118 * mm])]

story += [p("1. Executive summary", "H1x")]
story += [p(
    "The original underwear estimator accepted a size only when <b>four calibrated flat-garment "
    "measurements</b> - front rise, full length, hips, and waist - were all within tolerance of "
    "exactly one supplied size reference. The waist_only branch changes that decision rule so "
    "that <b>only the calibrated flat waist width</b> determines M, L, XL, or 2XL.")]
story += [p(
    "The change is intentionally narrow. Segmentation, landmark detection, inspection mask checks, "
    "landmark geometry checks, calibration, units, camera capture, concurrency protection, snapshots, "
    "and the full reference table remain in place.", "Callout")]
story += [table([
    ["Question", "Original", "waist_only"],
    ["Required sizing inputs", "4 measurements", "Waist only"],
    ["Decision distance", "RMSE across 4 deltas", "Absolute waist delta"],
    ["Tolerance pass", "Every delta within +/-0.50 in", "Waist delta within +/-0.50 in"],
    ["Ambiguity", "Multiple full-vector matches", "Overlapping waist intervals"],
    ["Non-waist measurements", "Affect the result", "Ignored by estimate_size()"],
    ["Unknown-size handling", "Uncertain", "Uncertain"],
], widths=[42 * mm, 62 * mm, 62 * mm])]
story += [PageBreak()]

story += [p("2. Baseline and scope", "H1x")]
story += [p("2.1 Baseline document", "H2x")]
story += [p(
    "The comparison baseline is <b>docs/CODEBASE_GUIDE.pdf</b>, especially Section 8.4, "
    "which documents the original four-dimensional <font name='Courier'>estimate_size()</font> "
    "algorithm. The baseline reference values are measured laid-flat garment dimensions in inches, "
    "not body measurements and not an industry-wide sizing chart.")]
story += [p("2.2 In scope", "H2x")]
story += bullets([
    "The size-decision function in src/garmentiq/underwear_sizes.py.",
    "Tests that define the sizing contract in test/test_underwear.py.",
    "Operator wording in phone_web/index.html.",
    "Sizing descriptions in docs/LIVE_PRODUCTION_LINE.md and docs/CODEBASE_GUIDE.md.",
])
story += [p("2.3 Explicitly out of scope", "H2x")]
story += bullets([
    "No model retraining and no checkpoint changes.",
    "No landmark schema changes; shorts landmarks are still used for underwear.",
    "No change to camera capture, stillness detection, ROI cropping, or TLS hosting.",
    "No change to ruler/reference calibration or pixel-to-inch conversion.",
    "No change to frame rejection heuristics or request locking.",
    "No removal of non-waist values from REFERENCES or measurement responses.",
])
story += [p(
    "Important distinction: the application may continue to <b>measure and display</b> front rise, "
    "full length, and hips. They simply no longer influence the size returned by estimate_size().",
    "Callout",
)]

story += [p("3. Behavioral contract", "H1x")]
story += [p("3.1 Required input", "H2x")]
story += [p(
    "The function now reads only <font name='Courier'>measurements['waist']</font>. Its distance must "
    "be a finite positive Python int or float, must not be bool, and its unit must be exactly "
    "<font name='Courier'>'in'</font>. Missing or invalid waist data produces an uncertain response.")]
story += [p("3.2 Ignored input", "H2x")]
story += [p(
    "Any values under <font name='Courier'>'front rise'</font>, <font name='Courier'>'full length'</font>, "
    "or <font name='Courier'>'hips'</font> are ignored by the estimator. They may be missing, invalid, "
    "or far from the reference without changing the size decision, provided waist remains valid.")]
story += [p("3.3 Output compatibility", "H2x")]
story += [p(
    "The response keys remain compatible: status, size, reason, tolerance_in, candidates, reference, "
    "closest_size, closest_reference, closest_deltas_in, and closest_rmse_in. The semantic difference "
    "is that each candidate's deltas_in now contains only waist, and rmse_in is the absolute waist "
    "delta rather than a four-dimensional RMSE.")]
story += [p("3.4 Conservative acceptance remains", "H2x")]
story += [p(
    "A nearest reference is still not automatically accepted. Exactly one reference must be within "
    "tolerance. Zero matches and multiple matches remain uncertain and require manual review.")]
story += [PageBreak()]

story += [p("4. Algorithm comparison", "H1x")]
story += [p("4.1 Original four-measurement algorithm", "H2x")]
story += [p(
    "For every expected measurement name, validate inches and a positive finite distance. For each "
    "size, calculate four deltas, calculate their RMSE, and accept the candidate only when every "
    "absolute delta is at most the tolerance.")]
story += [p(
    "for name in ('front rise', 'full length', 'hips', 'waist'):\n"
    "    validate(measurements[name])\n"
    "for size, reference in REFERENCES.items():\n"
    "    deltas = values - reference\n"
    "    rmse = sqrt(mean(delta ** 2))\n"
    "    within = all(abs(delta) <= tolerance for delta in deltas)", "Codex")]
story += [p("4.2 Waist-only algorithm", "H2x")]
story += [p(
    "Validate only the waist measurement. For each size, subtract the reference waist. The absolute "
    "delta is both the comparison distance and the tolerance test.")]
story += [p(
    "waist = validate(measurements['waist'])\n"
    "for size, reference in REFERENCES.items():\n"
    "    delta = waist - reference['waist']\n"
    "    distance = abs(delta)\n"
    "    within = abs(delta) <= tolerance", "Codex")]
story += [p("4.3 Selection rule", "H2x")]
story += [table([
    ["Number of candidates within tolerance", "Result"],
    ["0", "status=uncertain; size=None; closest candidate still reported"],
    ["1", "status=matched; size=that candidate"],
    ["2 or more", "status=uncertain; size=None; manual review required"],
], widths=[67 * mm, 99 * mm])]

story += [p("5. Waist references and tolerance geometry", "H1x")]
story += [p(
    "The branch retains the supplied flat-waist references and the provisional +/-0.50 inch "
    "allowance. The following intervals are inclusive.")]
story += [table([
    ["Size", "Reference waist (in)", "Accepted interval (in)"],
    ["M", "13.69", "13.19 to 14.19"],
    ["L", "14.62", "14.12 to 15.12"],
    ["XL", "15.79", "15.29 to 16.29"],
    ["2XL", "16.84", "16.34 to 17.34"],
], widths=[38 * mm, 60 * mm, 68 * mm])]
story += [p("5.1 Ambiguous M/L overlap", "H2x")]
story += [p(
    "M and L overlap from 14.12 through 14.19 inches. A waist inside that interval matches both "
    "references and therefore returns uncertain. This is expected and prevents a forced label.")]
story += [p("5.2 Unmatched gaps", "H2x")]
story += [p(
    "There is a 0.17-inch gap between L and XL (15.12 to 15.29) and a 0.05-inch gap between XL "
    "and 2XL (16.29 to 16.34). Values strictly inside those gaps match no reference and return uncertain.")]
story += [p("5.3 Why the boundary test changed", "H2x")]
story += [p(
    "The old M + 0.50 inch test becomes ambiguous because 14.19 inches falls inside both M and L. "
    "The test now uses M - 0.50 inches (13.19), which is an unambiguous inclusive boundary; one "
    "additional hundredth (13.18) correctly becomes uncertain.")]

story += [p("6. File-by-file change inventory", "H1x")]
story += [table([
    ["File", "Change", "Runtime effect"],
    ["src/garmentiq/underwear_sizes.py", "Added SIZE_MEASUREMENT='waist'; validates one input; compares one delta; updated reasons/docstring", "Waist alone decides size"],
    ["test/test_underwear.py", "Replaced four-dimension expectations with waist-only contract tests; moved invalid-input checks to waist", "Prevents regression to four-factor sizing"],
    ["phone_web/index.html", "Changed operator explanation from all four dimensions to flat waist", "UI matches server behavior"],
    ["phone_server.py", "Updated one explanatory comment to describe waist-only ambiguity", "No runtime behavior change"],
    ["docs/LIVE_PRODUCTION_LINE.md", "Updated acceptance and uncertainty wording", "Operating guide matches branch"],
    ["docs/CODEBASE_GUIDE.md", "Updated estimator steps and uncertainty definition", "Technical baseline is no longer misleading"],
    ["docs/WAIST_ONLY_CHANGES.pdf", "New complete comparison document", "Review and handover artifact"],
], widths=[47 * mm, 76 * mm, 43 * mm], font_size=7.6)]
story += [p("6.1 Reference table deliberately retained", "H2x")]
story += [p(
    "REFERENCES still stores front rise, full length, hips, and waist. This preserves the supplied "
    "sample record, the UI reference table, snapshot comparisons, and future ability to restore "
    "multi-factor sizing. Only the estimator's decision vector changed.")]
story += [p("6.2 Response field names deliberately retained", "H2x")]
story += [p(
    "The field name rmse_in remains for API compatibility even though, for a one-dimensional vector, "
    "RMSE is mathematically equal to absolute error. Renaming it would require unnecessary frontend "
    "and consumer changes.")]

story += [PageBreak()]
story += [p("7. What did not change in the processing pipeline", "H1x")]
story += [table([
    ["Stage", "Unchanged behavior"],
    ["Phone capture", "Browser captures the selected ROI at native pixel resolution"],
    ["Calibration", "Known reference converts pixel distances to inches"],
    ["Segmentation", "BiRefNet produces the garment mask"],
    ["Landmark detection", "Shorts landmark layout supplies underwear points"],
    ["Frame safety", "Mask and landmark geometry can reject unsuitable frames"],
    ["Measurement", "LiveDetector may calculate all garment dimensions"],
    ["Concurrency", "One inference request at a time; busy requests receive 429"],
    ["Unknown handling", "Uncertain remains a normal response, not a server error"],
], widths=[47 * mm, 119 * mm])]
story += [p(
    "The waist-only change occurs after calibrated measurements exist. It does not make calibration "
    "optional, and it does not turn pixel width into inches by itself.", "Callout")]
story += [PageBreak()]

story += [p("8. Verification and acceptance evidence", "H1x")]
story += [p("8.1 Automated suite", "H2x")]
story += [p(
    "The underwear and HTTP integration suite completed successfully: <b>21 tests passed</b>. It "
    "covers exact reference sizes, tolerance boundaries, ambiguous results, invalid waist values, "
    "closest-size reporting, frame rejection, real measurement math, API validation, concurrency, "
    "error recovery, reference delivery, and frontend assets.")]
story += [p("8.2 Contract-specific checks", "H2x")]
story += bullets([
    "Changing full length to an extreme value no longer changes an XL waist match.",
    "Invalid or missing waist still returns uncertain.",
    "A calibrated 14.62-inch waist still resolves to L in the integration fixture.",
    "The L/XL midpoint under a widened tolerance is ambiguous and is not forced.",
    "API response shape remains compatible with the existing phone server.",
])
story += [p("8.3 Suggested production acceptance test", "H2x")]
story += [table([
    ["Step", "Expected observation"],
    ["Calibrate against a known reference", "Scale is finite and expressed as pixels per inch"],
    ["Measure one physical sample per size", "Waist is close to its supplied reference"],
    ["Repeat each sample 10 times", "Variation is recorded; ambiguous runs remain uncertain"],
    ["Test overlap/gap values", "No forced result where two or zero references match"],
    ["Move or resize camera", "Recalibration is required before trusting inches"],
], widths=[65 * mm, 101 * mm])]

story += [p("9. Risks, limitations, and operating implications", "H1x")]
story += [p("9.1 Reduced evidence", "H2x")]
story += [p(
    "A single width cannot distinguish garments that share a waistband width but differ in rise, hip "
    "shape, stretch, cut, or labeling. The branch intentionally trades discrimination for a simpler rule.")]
story += [p("9.2 Flat width is not body circumference", "H2x")]
story += [p(
    "The waist value is the laid-flat garment width. It is not automatically a body-waist value, "
    "a doubled circumference, or a universal retail size. Elastic tension and measurement procedure matter.")]
story += [p("9.3 Calibration remains decisive", "H2x")]
story += [p(
    "Scale error directly shifts the only deciding feature. With four features, one noisy dimension "
    "could cause rejection; with one feature, a biased waist can cause a wrong match. Keep the camera "
    "fixed, keep garment and reference in the same plane, and recalibrate after geometry changes.")]
story += [p("9.4 Tolerance is provisional", "H2x")]
story += [p(
    "+/-0.50 inches is not a validated manufacturing tolerance or statistical confidence interval. "
    "Do not widen it merely to reduce uncertain outcomes; wider bands increase overlap and ambiguity.")]
story += [p("9.5 Recommended guardrails", "H2x")]
story += bullets([
    "Continue using InspectionRejected gates before size estimation.",
    "Show uncertainty to the operator rather than silently choosing closest_size.",
    "Record measured waist, delta, calibration, and image for audits.",
    "Validate references against real production samples before operational use.",
    "Treat a product/style change as a new reference-validation exercise.",
])

story += [p("10. Migration, deployment, and rollback", "H1x")]
story += [p("10.1 Deploying the branch", "H2x")]
story += [p(
    "Fetch the repository and check out <font name='Courier'>waist_only</font>. Install the same "
    "runtime and model checkpoints used by main. No database migration, checkpoint conversion, or "
    "frontend build step is introduced by this change.")]
story += [p(
    "git fetch origin\n"
    "git switch waist_only\n"
    "python -m unittest discover -s test -p test_underwear.py -v", "Codex")]
story += [p("10.2 Rollback", "H2x")]
story += [p(
    "Switch back to main to restore the original four-measurement estimator. Because the reference "
    "table and API response shape were preserved, rollback does not require data conversion.")]
story += [p(
    "git switch main", "Codex")]
story += [p("10.3 Review checklist", "H2x")]
story += bullets([
    "Confirm the active branch is waist_only.",
    "Confirm CUDA/CPU runtime and checkpoints load successfully.",
    "Run the 21-test underwear suite.",
    "Verify the phone UI states that waist alone determines size.",
    "Perform physical repeatability testing before production decisions.",
])

story += [p("Appendix A - Example inputs and outputs", "H1x")]
story += [p("A.1 Matched L", "H2x")]
story += [p(
    "Input:\n{'waist': {'distance': 14.62, 'unit': 'in'},\n"
    " 'hips': {'distance': 99.0, 'unit': 'in'}}\n\n"
    "Decision:\n{'status': 'matched', 'size': 'L',\n"
    " 'reason': 'Waist measurement matches one supplied reference.'}", "Codex")]
story += [p(
    "The deliberately incorrect hips value is ignored. This demonstrates the new contract; it is "
    "not a recommendation to suppress measurement-quality checks elsewhere.")]
story += [p("A.2 Ambiguous M/L", "H2x")]
story += [p(
    "Input waist: 14.15 in\n"
    "M delta: +0.46 in (within tolerance)\n"
    "L delta: -0.47 in (within tolerance)\n"
    "Result: uncertain because two candidates match", "Codex")]
story += [p("A.3 No match", "H2x")]
story += [p(
    "Input waist: 15.20 in\n"
    "L delta: +0.58 in\n"
    "XL delta: -0.59 in\n"
    "Result: uncertain because no candidate is within +/-0.50 in", "Codex")]
story += [p("A.4 Invalid calibration unit", "H2x")]
story += [p(
    "Input: {'waist': {'distance': 500, 'unit': 'px'}}\n"
    "Result: uncertain - waist must be calibrated to inches", "Codex")]

story += [p("Appendix B - Source references", "H1x")]
story += [table([
    ["Repository source", "Role in this guide"],
    ["docs/CODEBASE_GUIDE.pdf", "Original four-measurement design baseline"],
    ["src/garmentiq/underwear_sizes.py", "Sizing references, validation, and decision algorithm"],
    ["test/test_underwear.py", "Executable behavior contract and integration evidence"],
    ["phone_web/index.html", "Operator-facing explanation"],
    ["docs/LIVE_PRODUCTION_LINE.md", "Production operation and calibration guidance"],
], widths=[64 * mm, 102 * mm])]
story += [Spacer(1, 7 * mm), p(
    "End of document. This guide describes the waist_only branch only. It should be read together "
    "with the existing full codebase guide for architecture and module-level details.", "Callout")]


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
GuideDoc(str(OUTPUT)).build(story)
print(OUTPUT)
