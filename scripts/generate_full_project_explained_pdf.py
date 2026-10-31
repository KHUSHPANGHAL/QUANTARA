"""
QUANTARA AI — FULL PROJECT EXPLAINED PDF GENERATOR
Script: scripts/generate_full_project_explained_pdf.py
Description: Generates an institutional-grade, multi-page PDF summarizing the entire Quantara AI research project.
Outputs:
  - explanation/Quantara_AI_Full_Project_Explained.pdf
  - docs/explanation/Quantara_AI_Full_Project_Explained.pdf
  - docs/project_explainer/Quantara_AI_Full_Project_Explained.pdf
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH_EXPLANATION = ROOT / "explanation" / "Quantara_AI_Full_Project_Explained.pdf"
PDF_PATH_DOCS_EXP = ROOT / "docs" / "explanation" / "Quantara_AI_Full_Project_Explained.pdf"
PDF_PATH_DOCS_PROJECT = ROOT / "docs" / "project_explainer" / "Quantara_AI_Full_Project_Explained.pdf"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "Quantara AI — Institutional Quantitative Research Suite | Full Project Explained")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer (all pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_str)
        self.drawString(54, 30, "CONFIDENTIAL & PROPRIETARY — QUANTARA AI QUANTITATIVE AUDIT")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 42, letter[0] - 54, 42)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH_EXPLANATION),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    PRIMARY = colors.HexColor("#0f172a")      # Slate 900
    ACCENT = colors.HexColor("#0369a1")       # Sky 700
    TEAL = colors.HexColor("#0f766e")         # Teal 700
    DARK_TEXT = colors.HexColor("#1e293b")    # Slate 800
    MUTED = colors.HexColor("#64748b")        # Slate 500
    CARD_BG = colors.HexColor("#f8fafc")      # Slate 50
    BORDER_COLOR = colors.HexColor("#cbd5e1") # Slate 300
    WARNING_BG = colors.HexColor("#fef3c7")   # Amber 100
    WARNING_BORDER = colors.HexColor("#f59e0b")

    style_title = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=4,
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=ACCENT,
        spaceAfter=10,
    )
    style_h1 = ParagraphStyle(
        "DocH1",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=6,
    )
    style_h2 = ParagraphStyle(
        "DocH2",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=ACCENT,
        spaceBefore=8,
        spaceAfter=3,
    )
    style_body = ParagraphStyle(
        "DocBody",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT,
        spaceAfter=5,
    )
    style_body_bold = ParagraphStyle(
        "DocBodyBold",
        parent=style_body,
        fontName="Helvetica-Bold",
    )
    style_bullet = ParagraphStyle(
        "DocBullet",
        parent=style_body,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )
    style_table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
    )
    style_table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=style_table_cell,
        fontName="Helvetica-Bold",
        textColor=PRIMARY,
    )
    style_table_header = ParagraphStyle(
        "TableHeader",
        parent=style_table_cell_bold,
        textColor=colors.white,
    )
    style_warning = ParagraphStyle(
        "DocWarning",
        parent=style_body,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#92400e"),
    )

    story = []

    # Title Block
    story.append(Paragraph("QUANTARA AI — FULL PROJECT EXPLAINED", style_title))
    story.append(Paragraph("Comprehensive Institutional Research Suite: Machine Learning, Deep Learning, & Market Regime Economics", style_subtitle))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')} | <b>Universe:</b> 8 Liquid NSE Blue-Chips | <b>Research Scope:</b> 2017–2026", style_body))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceBefore=4, spaceAfter=8))

    # Warning Box
    warning_box = Table(
        [[Paragraph("<b>MANDATORY RESEARCH NOTICE:</b> Quantara AI is an experimental quantitative research platform. All findings represent historical backtest simulations. Past performance does not guarantee future results. <b>Live trading is strictly disabled; no broker APIs or order-routing mechanisms exist.</b>", style_warning)]],
        colWidths=[504],
    )
    warning_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARNING_BG),
        ("BOX", (0, 0), (-1, -1), 1, WARNING_BORDER),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(warning_box)
    story.append(Spacer(1, 8))

    # Section 1: Executive Overview
    story.append(Paragraph("1. Executive Overview & Research Objective", style_h1))
    story.append(Paragraph(
        "Quantara AI was created to rigorously evaluate whether predictive artificial intelligence can systematically forecast "
        "medium-term equity trends in the Indian stock market after factoring in authentic transaction costs and market friction. "
        "The project analyzes 8 highly liquid National Stock Exchange (NSE) securities: <b>BHARTIARTL, HDFCBANK, ICICIBANK, INFY, "
        "ITC, RELIANCE, SBIN, and TCS</b> spanning nearly a decade of authentic daily price history (2017–2026, 19,380+ observations). "
        "The primary prediction target evaluates the <b>20-trading-session forward return (~1 calendar month)</b>, executing orders at Open t+1 based on Close t signals.",
        style_body
    ))
    story.append(Spacer(1, 4))

    # Section 2: Research Chronology
    story.append(Paragraph("2. Chronological Research Journey (Phases 1 to 26)", style_h1))
    
    phases_summary = [
        ("Phases 1–4: Data Hygiene & Leakage Prevention",
         "Audited 19,776 raw records from zero-synthetic source data. Engineered 20 technical indicators. Established strict 20-day embargoes between chronological splits to eliminate forward-return leakage. Verified scalers fit only on historical training rows."),
        ("Phases 13–15B: The Phase 13 Machine Learning Baseline",
         "Standardized on 15 trend features (group_1_trend). Established the authoritative Phase 13 Logistic Regression baseline: 52.63% test balanced accuracy (47.28% standard accuracy, 46.32% macro F1). Confirmed that unconstrained long-only ML trading lost money (-17.46%) due to cumulative trading costs and bear market drawdowns."),
        ("Phases 16–19: Institutional Cost Models & Dividend Recovery",
         "Formally integrated India's statutory delivery fee schedule (0.3937% round-trip covering STT, exchange turnover, SEBI charges, stamp duty, GST, and brokerage). Extracted 124 authentic corporate dividend events to credit cash on ex-dates."),
        ("Phases 20–23: Breakthrough — Dynamic Cash Allocation (Rules D, E, F)",
         "Investigated market-regime filtering. Discovered that capital preservation is driven by dynamically holding cash during hostile regimes rather than aggressive trading. Rule F (dynamic regime allocation) protected capital by holding up to 100% cash in market downturns. Rules D, E, and F were mathematically audited and frozen."),
        ("Phases 24–25: Deep Learning Implementation & Forensic Audit",
         "Implemented 2-layer LSTM (56,162 params) and 2-layer GRU (42,658 params) using 60-day sequences. Phase 25 forensic audit identified two critical defects in Phase 24: (1) An invalid majority-baseline log loss of 21.9177 caused by passing hard labels [0, 1] into scikit-learn, and (2) Walk-forward folds 1–3 were evaluated on static models without retraining (in-sample training memorization)."),
        ("Phase 26: Definitive Corrective Run & True Walk-Forward Retraining",
         "Corrected the baseline probability bug (valid log loss = 0.7794). Executed true walk-forward retraining across 4 distinct expanding folds (4,647 out-of-sample observations). Proved that recurrent deep learning (LSTM: 49.92%, GRU: 48.68%) did NOT outperform simple Logistic Regression (50.07%)."),
    ]

    for p_title, p_desc in phases_summary:
        story.append(Paragraph(f"<b>• {p_title}:</b> {p_desc}", style_body))

    story.append(Spacer(1, 6))

    # Section 3: Definitive Empirical Benchmark Table
    story.append(Paragraph("3. Definitive Out-of-Sample Performance Comparison", style_h1))
    story.append(Paragraph(
        "Below is the verified performance across <b>4,647 genuine out-of-sample observations</b> (2023–2026) evaluated under true walk-forward retraining with 20-day block bootstrap confidence intervals:",
        style_body
    ))

    benchmark_data = [
        [Paragraph("Model Architecture", style_table_header),
         Paragraph("Balanced Acc", style_table_header),
         Paragraph("95% Block Bootstrap CI", style_table_header),
         Paragraph("Macro F1", style_table_header),
         Paragraph("Log Loss", style_table_header),
         Paragraph("Brier Score", style_table_header),
         Paragraph("Paired Diff vs LR", style_table_header)],
        [Paragraph("<b>Corrected Majority Baseline</b>", style_table_cell),
         Paragraph("50.00%", style_table_cell),
         Paragraph("[50.00%, 50.00%]", style_table_cell),
         Paragraph("35.75%", style_table_cell),
         Paragraph("0.6977", style_table_cell),
         Paragraph("0.2521", style_table_cell),
         Paragraph("N/A", style_table_cell)],
        [Paragraph("<b>Logistic Regression (Retrained)</b>", style_table_cell),
         Paragraph("<b>50.07%</b>", style_table_cell_bold),
         Paragraph("[46.19%, 54.25%]", style_table_cell),
         Paragraph("<b>49.91%</b>", style_table_cell_bold),
         Paragraph("<b>0.6942</b>", style_table_cell_bold),
         Paragraph("<b>0.2505</b>", style_table_cell_bold),
         Paragraph("Baseline", style_table_cell)],
        [Paragraph("<b>LSTM Classifier (Fresh Retrained)</b>", style_table_cell),
         Paragraph("49.92%", style_table_cell),
         Paragraph("[44.96%, 55.11%]", style_table_cell),
         Paragraph("46.89%", style_table_cell),
         Paragraph("0.6983", style_table_cell),
         Paragraph("0.2524", style_table_cell),
         Paragraph("-0.15% (CI spans 0)", style_table_cell)],
        [Paragraph("<b>GRU Classifier (Fresh Retrained)</b>", style_table_cell),
         Paragraph("48.68%", style_table_cell),
         Paragraph("[45.73%, 51.44%]", style_table_cell),
         Paragraph("39.37%", style_table_cell),
         Paragraph("0.6992", style_table_cell),
         Paragraph("0.2528", style_table_cell),
         Paragraph("-1.39% (CI spans 0)", style_table_cell)],
    ]

    table_bm = Table(benchmark_data, colWidths=[120, 55, 95, 45, 45, 45, 99])
    table_bm.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CARD_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table_bm)
    story.append(Spacer(1, 8))

    # Section 4: Key Insights & Takeaways
    story.append(Paragraph("4. Core Scientific Takeaways", style_h1))
    
    insights = [
        "<b>Complexity Does Not Yield Alpha:</b> Moving from a 16-parameter linear model to a 56,162-parameter recurrent neural network did not improve predictive performance. Financial markets exhibit high noise and non-stationary distribution shifts that easily cause complex models to overfit spurious past sequences.",
        "<b>Mode Collapse Vulnerability:</b> Freshly retrained LSTM and GRU models suffered from persistent mode collapse toward the historical training class ('UP'), producing UP recall of 79%–93% while DOWN recall collapsed to 6%–21%. When market conditions turned sharply negative (2025–2026 downtrend), deep networks failed to adapt.",
        "<b>Risk Management Dominates Prediction:</b> Phases 20–23 confirmed that risk-adjusted capital preservation stems from dynamic cash management (Rule F regime filters) rather than trying to predict low-conviction directional price moves.",
        "<b>True Walk-Forward Evaluation is Crucial:</b> Testing models on data they were already trained on created an illusion of 55%–60% accuracy in Phase 24. Only true chronological retraining revealed actual out-of-sample performance (~50%).",
    ]

    for ins in insights:
        story.append(Paragraph(f"• {ins}", style_body))

    story.append(Spacer(1, 6))

    # Section 5: Next Steps (Phase 27)
    story.append(Paragraph("5. Planned Roadmap: Phase 27 Hypothetical Rs. 1,00,000 Backtest", style_h1))
    story.append(Paragraph(
        "The next planned research phase (Phase 27) will calculate the actual hypothetical rupee profit or loss from investing "
        "Rs. 1,00,000 starting capital using an <b>unfiltered 3-class target</b> (DOWN: < -2%, HOLD: [-2%, +2%], UP: > +2%). "
        "Because future outcomes cannot be filtered out in live trading, Phase 27 will simulate long-only execution (Predict UP: enter equity; "
        "Predict HOLD/DOWN: hold cash) with equal max 12.5% allocation across the 8 stocks, full statutory friction (0.3937% delivery fees + 0.05% slippage), "
        "dividend credits, and comparisons against Buy-and-Hold and Idle Cash benchmarks.",
        style_body
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF at: {PDF_PATH_EXPLANATION}")

    # Copy to docs/explanation/ and docs/project_explainer/
    shutil.copy2(PDF_PATH_EXPLANATION, PDF_PATH_DOCS_EXP)
    shutil.copy2(PDF_PATH_EXPLANATION, PDF_PATH_DOCS_PROJECT)
    print(f"Copied PDF to: {PDF_PATH_DOCS_EXP}")
    print(f"Copied PDF to: {PDF_PATH_DOCS_PROJECT}")


if __name__ == "__main__":
    build_pdf()
