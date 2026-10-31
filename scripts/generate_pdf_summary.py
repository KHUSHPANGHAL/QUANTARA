"""
Quantara AI — Executive Summary PDF Generator
Generates a concise, institutional-grade PDF report summarizing:
- Project Background & Architecture
- Phase 1 Forensic Audit & Data Hygiene
- Phase 2 Machine Learning Benchmark (5-Day Horizon)
- Financial Backtesting & Trading Realities
- Diagnosis of Model Accuracy & Next Steps
"""

import sys
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
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
PDF_PATH = ROOT / "Quantara_AI_Executive_Summary.pdf"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
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
            self.drawString(54, letter[1] - 36, "Quantara AI — Audited Data Preparation & Model Suite | Executive Summary")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer (all pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_str)
        self.drawString(54, 30, "CONFIDENTIAL & PROPRIETARY — QUANTARA AI PROJECT")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 42, letter[0] - 54, 42)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#0f172a")     # Slate 900
    ACCENT = colors.HexColor("#0369a1")      # Sky 700
    TEAL = colors.HexColor("#0f766e")        # Teal 700
    DARK_TEXT = colors.HexColor("#1e293b")   # Slate 800
    MUTED = colors.HexColor("#64748b")       # Slate 500
    BG_LIGHT = colors.HexColor("#f8fafc")    # Slate 50
    CARD_BG = colors.HexColor("#f1f5f9")     # Slate 100
    BORDER_COLOR = colors.HexColor("#cbd5e1")# Slate 300
    SUCCESS_TEXT = colors.HexColor("#166534")# Green 800
    WARNING_BG = colors.HexColor("#fef3c7")  # Amber 100
    WARNING_BORDER = colors.HexColor("#f59e0b")

    # Typography Styles
    style_title = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY,
        spaceAfter=4,
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=ACCENT,
        spaceAfter=12,
    )
    style_h1 = ParagraphStyle(
        "DocH1",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
    )
    style_h2 = ParagraphStyle(
        "DocH2",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=ACCENT,
        spaceBefore=8,
        spaceAfter=4,
    )
    style_body = ParagraphStyle(
        "DocBody",
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=DARK_TEXT,
        spaceAfter=6,
    )
    style_body_bold = ParagraphStyle(
        "DocBodyBold",
        parent=style_body,
        fontName="Helvetica-Bold",
    )
    style_bullet = ParagraphStyle(
        "DocBullet",
        parent=style_body,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
    )
    style_callout = ParagraphStyle(
        "DocCallout",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT,
    )
    style_table_header = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1, # Center
    )
    style_table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=DARK_TEXT,
        alignment=1,
    )
    style_table_cell_left = ParagraphStyle(
        "TableCellLeft",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=DARK_TEXT,
        alignment=0,
    )

    story = []

    # ---------------------------------------------------------
    # Header Banner
    # ---------------------------------------------------------
    story.append(Paragraph("Quantara AI — Audited Data Preparation & Model Suite", style_title))
    story.append(Paragraph("Comprehensive Forensic Audit, Pipeline Architecture & Machine Learning Validation Report", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceBefore=0, spaceAfter=10))

    # Meta Info Card Table
    meta_data = [
        [
            Paragraph("<b>Target Asset Class:</b> National Stock Exchange (NSE) Equities", style_body),
            Paragraph("<b>Time Span:</b> 10 Years (2016-10-24 → 2026-10-23)", style_body),
        ],
        [
            Paragraph("<b>Total Raw Observations:</b> 19,776 Sessions (8 Stocks)", style_body),
            Paragraph("<b>Integrity Status:</b> 75/75 Checks Passed (SHA-256 Verified)", style_body),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 1: Executive Overview
    # ---------------------------------------------------------
    story.append(Paragraph("1. Executive Overview & Philosophy", style_h1))
    story.append(Paragraph(
        "This project establishes an institutional-grade machine learning data preparation and validation pipeline "
        "rebuilt from historical NSE equity data. Unlike standard academic demonstrations that suffer from lookahead bias "
        "or arbitrary price memorization, this repository enforces strict data stationarity, cryptographic tracking, "
        "and leak-free chronological boundary purging across <b>8 core Indian stocks</b>: <i>BHARTIARTL, HDFCBANK, "
        "ICICIBANK, INFY, ITC, RELIANCE, SBIN, and TCS</i>.",
        style_body
    ))
    story.append(Paragraph("<b>Three Core Governing Rules:</b>", style_body_bold))
    story.append(Paragraph("• <b>Zero Synthetic Data:</b> Not a single price bar has been hallucinated, interpolated, or artificially generated.", style_bullet))
    story.append(Paragraph("• <b>Strict Stationarity:</b> Raw price levels (e.g. ₹2,500) are eliminated from inputs; all features are converted to dimensionless ratios.", style_bullet))
    story.append(Paragraph("• <b>Boundary Purging:</b> Samples whose forward prediction window overlaps subsequent splits are quarantined to guarantee zero lookahead leakage.", style_bullet))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # Section 2: Phase 1 Forensic Audit (What Was Fixed)
    # ---------------------------------------------------------
    story.append(Paragraph("2. Phase 1 Forensic Audit: Correcting the Baseline", style_h1))
    story.append(Paragraph(
        "A forensic review of the original uploaded snapshot (<code>source_snapshot/</code>) revealed 4 fatal quantitative design flaws, "
        "which were completely corrected in the audited pipeline (<code>data/splits/</code>):",
        style_body
    ))

    flaw_table_data = [
        [
            Paragraph("Design Dimension", style_table_header),
            Paragraph("Original Baseline Flaw", style_table_header),
            Paragraph("Audited Pipeline Correction", style_table_header),
        ],
        [
            Paragraph("<b>Model Features</b>", style_table_cell_left),
            Paragraph("Fed raw prices (Open, High, Low, Close, Bollinger Bands) directly into models.", style_table_cell_left),
            Paragraph("Engineered <b>20 stationary features</b> (returns, moving average divergences, RSI, MACD %, volume differences).", style_table_cell_left),
        ],
        [
            Paragraph("<b>Dataset Splits</b>", style_table_cell_left),
            Paragraph("Only 2 splits (80% Train, 20% Test); no validation set.", style_table_cell_left),
            Paragraph("<b>3 Chronological Splits:</b> 80% Train (to Oct 2024), 10% Validation (to Oct 2025), 10% Test (to Oct 2026).", style_table_cell_left),
        ],
        [
            Paragraph("<b>Boundary Leakage</b>", style_table_cell_left),
            Paragraph("No purging. Late training rows predicted returns overlapping into the test period.", style_table_cell_left),
            Paragraph("<b>Purged boundary rows:</b> 16 (1d), 80 (5d), and 320 (20d) overlapping rows quarantined into <code>purged_boundary_rows.csv</code>.", style_table_cell_left),
        ],
        [
            Paragraph("<b>Prediction Horizons</b>", style_table_cell_left),
            Paragraph("Hardcoded to a single 1-day timeframe.", style_table_cell_left),
            Paragraph("Three distinct timeframes: <b>1-Day</b> (daily), <b>5-Day</b> (weekly), and <b>20-Day</b> (monthly).", style_table_cell_left),
        ],
    ]
    flaw_table = Table(flaw_table_data, colWidths=[100, 190, 214])
    flaw_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(flaw_table)
    story.append(Spacer(1, 10))

    # Page Break for clean multi-page layout
    story.append(PageBreak())

    # ---------------------------------------------------------
    # Section 3: Phase 2 Model Development & Benchmarking
    # ---------------------------------------------------------
    story.append(Paragraph("3. Phase 2 Machine Learning Suite (5-Day Horizon)", style_h1))
    story.append(Paragraph(
        "Focusing on the <b>5-Day Forward Return Horizon (`horizon_5d`)</b>, four distinct model architectures were trained and "
        "benchmarked side-by-side on the untouched Test Split (<b>1,944 out-of-sample trading sessions</b> from late 2025 to late 2026). "
        "The classification target has 3 classes: <b>`DOWN`</b> (&lt; -1.0%), <b>`HOLD`</b> ([-1%, +1%]), and <b>`UP`</b> (&gt; +1.0%).",
        style_body
    ))

    perf_table_data = [
        [
            Paragraph("Model Architecture", style_table_header),
            Paragraph("Val Bal Acc", style_table_header),
            Paragraph("Test Accuracy", style_table_header),
            Paragraph("Test Bal Acc", style_table_header),
            Paragraph("Macro F1", style_table_header),
            Paragraph("DOWN Prec.", style_table_header),
            Paragraph("UP Prec.", style_table_header),
        ],
        [
            Paragraph("<b>0. Dummy Baseline</b> (Chance)", style_table_cell_left),
            Paragraph("33.33%", style_table_cell),
            Paragraph("31.48%", style_table_cell),
            Paragraph("33.33%", style_table_cell),
            Paragraph("0.1596", style_table_cell),
            Paragraph("0.0%", style_table_cell),
            Paragraph("31.5%", style_table_cell),
        ],
        [
            Paragraph("<b>1. Logistic Regression</b>", style_table_cell_left),
            Paragraph("34.69%", style_table_cell),
            Paragraph("<b>37.65%</b>", style_table_cell),
            Paragraph("<b>40.73%</b>", style_table_cell),
            Paragraph("0.3557", style_table_cell),
            Paragraph("<b>55.1%</b>", style_table_cell),
            Paragraph("<b>36.3%</b>", style_table_cell),
        ],
        [
            Paragraph("<b>2. Random Forest</b> (100 Trees)", style_table_cell_left),
            Paragraph("34.18%", style_table_cell),
            Paragraph("37.09%", style_table_cell),
            Paragraph("38.56%", style_table_cell),
            Paragraph("<b>0.3687</b>", style_table_cell),
            Paragraph("53.6%", style_table_cell),
            Paragraph("32.0%", style_table_cell),
        ],
        [
            Paragraph("<b>3. LightGBM</b> (Tuned Trees)", style_table_cell_left),
            Paragraph("<b>34.71%</b>", style_table_cell),
            Paragraph("33.80%", style_table_cell),
            Paragraph("34.97%", style_table_cell),
            Paragraph("0.3374", style_table_cell),
            Paragraph("45.8%", style_table_cell),
            Paragraph("29.4%", style_table_cell),
        ],
    ]
    perf_table = Table(perf_table_data, colWidths=[130, 60, 62, 62, 55, 65, 70])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 4: Financial Simulation (Backtest with Friction)
    # ---------------------------------------------------------
    story.append(Paragraph("4. Financial Strategy Simulation (Backtest with Real Friction)", style_h1))
    story.append(Paragraph(
        "A quantitative trading simulation was executed across all 1,944 test sessions, applying realistic Indian market round-trip "
        "friction (<b>0.15% transaction cost</b> for brokerage, STT tax, and exchange slippage):",
        style_body
    ))

    fin_table_data = [
        [
            Paragraph("Strategy Dimension", style_table_header),
            Paragraph("Buy-and-Hold Market Benchmark", style_table_header),
            Paragraph("Logistic Regression Model", style_table_header),
            Paragraph("LightGBM Classifier", style_table_header),
        ],
        [
            Paragraph("<b>Underlying Market Return</b>", style_table_cell_left),
            Paragraph("-0.29% gross (-0.44% net)", style_table_cell),
            Paragraph("-0.29% gross (-0.44% net)", style_table_cell),
            Paragraph("-0.29% gross (-0.44% net)", style_table_cell),
        ],
        [
            Paragraph("<b>Long Trades Taken (`UP`)</b>", style_table_cell_left),
            Paragraph("1,944 sessions", style_table_cell),
            Paragraph("791 trades", style_table_cell),
            Paragraph("756 trades", style_table_cell),
        ],
        [
            Paragraph("<b>Long Strategy Net Return</b>", style_table_cell_left),
            Paragraph("-0.44%", style_table_cell),
            Paragraph("<b>-0.20% (+0.25% Net Alpha)</b>", style_table_cell),
            Paragraph("-0.53% (-0.09% Net Alpha)", style_table_cell),
        ],
        [
            Paragraph("<b>Downside Short/Hedge Signals</b>", style_table_cell_left),
            Paragraph("0 (Always long)", style_table_cell),
            Paragraph("176 signals (<b>64.2% drop rate</b>)", style_table_cell),
            Paragraph("415 signals (<b>57.1% drop rate</b>)", style_table_cell),
        ],
        [
            Paragraph("<b>Downside Short Net Profit</b>", style_table_cell_left),
            Paragraph("N/A", style_table_cell),
            Paragraph("<b>+0.79% Net Profit per Trade</b>", style_table_cell),
            Paragraph("<b>+0.25% Net Profit per Trade</b>", style_table_cell),
        ],
    ]
    fin_table = Table(fin_table_data, colWidths=[130, 124, 125, 125])
    fin_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TEAL),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(fin_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 5: The "Problem" Diagnosis (Why ~38% Accuracy?)
    # ---------------------------------------------------------
    story.append(Paragraph("5. Forensic Diagnosis: Why Is Model Accuracy ~38%?", style_h1))
    story.append(Paragraph(
        "A common beginner misconception is expecting 80% to 90% accuracy in stock forecasting. In institutional finance, "
        "market moves contain heavy noise. <b>38% balanced accuracy in a 3-class problem represents a true +5% edge over random guessing (33.3%)</b>. "
        "The performance ceiling in this dataset is dictated by four structural realities:",
        style_body
    ))

    diag_box_data = [
        [
            Paragraph("<b>1. Severe Market Regime Shift:</b> The training split (2017–2024) was a 7-year bull market (41% UP). The test split (2025–2026) was a market correction (40% DOWN). Tree models memorized bull breakout patterns that turned into bull traps during the correction.", style_body),
        ],
        [
            Paragraph("<b>2. Information Starvation:</b> The model relies exclusively on 20 basic technical indicators. It is blind to the NIFTY 50 benchmark index trend, RBI interest rates, oil prices, quarterly corporate earnings, and macroeconomic news.", style_body),
        ],
        [
            Paragraph("<b>3. The Choppy 'HOLD' Trap:</b> Small fractional moves (+0.95% vs +1.05%) arbitrarily switch labels between HOLD and UP, forcing the model to expend capacity trying to separate random daily market noise.", style_body),
        ],
        [
            Paragraph("<b>4. Stock Volatility Mismatch:</b> A flat ±1.0% threshold applies equally to low-volatility utility stocks (TCS) and high-beta cyclicals (SBIN), diluting signal uniformity across sectors.", style_body),
        ],
    ]
    diag_table = Table(diag_box_data, colWidths=[504])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), WARNING_BG),
        ('BOX', (0,0), (-1,-1), 1, WARNING_BORDER),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 10))

    # Page Break for clean conclusion & deliverables
    # ---------------------------------------------------------
    # Section 6: Top Feature Importance & Stock Breakdown
    # ---------------------------------------------------------
    story.append(Paragraph("6. Feature Importance & Stock-by-Stock Alpha", style_h1))
    
    col_feat = [
        Paragraph("<b>Top Technical Predictors:</b>", style_body_bold),
        Paragraph("1. <b>volatility_20d_annualized:</b> Regime gauge (quiet vs wild).", style_bullet),
        Paragraph("2. <b>intraday_range_pct:</b> Candle bar expansion precedes swings.", style_bullet),
        Paragraph("3. <b>bollinger_width_pct:</b> Volatility squeeze indicator.", style_bullet),
        Paragraph("4. <b>macd_histogram_pct:</b> Momentum acceleration.", style_bullet),
        Paragraph("5. <b>macd_pct:</b> Medium-term trend direction.", style_bullet),
        Paragraph("<i>Least Useful:</i> Daily 1-day noise (<code>daily_return_1d</code>).", style_bullet),
    ]
    
    col_stock = [
        Paragraph("<b>Stock-Specific Alpha (Test Set):</b>", style_body_bold),
        Paragraph("• <b>TCS.NS:</b> <b>41.56% Accuracy</b>, <b>+0.57% Alpha</b> (Best)", style_bullet),
        Paragraph("• <b>INFY.NS:</b> <b>35.80% Accuracy</b>, <b>+0.26% Alpha</b> (Strong)", style_bullet),
        Paragraph("• <b>BHARTIARTL.NS:</b> 30.45% Accuracy, +0.10% Alpha", style_bullet),
        Paragraph("• <b>HDFCBANK.NS:</b> 30.04% Accuracy, +0.07% Alpha", style_bullet),
        Paragraph("• <b>RELIANCE.NS:</b> 30.45% Accuracy, -0.40% Alpha", style_bullet),
        Paragraph("• <b>ITC.NS:</b> 34.16% Accuracy, -0.50% Alpha (Weakest)", style_bullet),
    ]
    
    stock_feat_table = Table([[col_feat, col_stock]], colWidths=[250, 254])
    stock_feat_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(stock_feat_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 7: Delivered Artifacts & How to Run
    # ---------------------------------------------------------
    story.append(Paragraph("7. Project Deliverables & User Tools", style_h1))
    story.append(Paragraph(
        "All trained models, evaluation metrics, and interactive tools are saved and fully functional in your project directory:",
        style_body
    ))

    deliverables_data = [
        [
            Paragraph("Deliverable Artifact", style_table_header),
            Paragraph("Location Path", style_table_header),
            Paragraph("Description / Function", style_table_header),
        ],
        [
            Paragraph("<b>Best Linear Model</b>", style_table_cell_left),
            Paragraph("<code>models/horizon_5d/logistic_regression_baseline.joblib</code>", style_table_cell_left),
            Paragraph("Standardized multinomial classifier. Highest balanced accuracy (40.7%) and short profit (+0.79%).", style_table_cell_left),
        ],
        [
            Paragraph("<b>Tree Model</b>", style_table_cell_left),
            Paragraph("<code>models/horizon_5d/lightgbm_5d_classifier.joblib</code>", style_table_cell_left),
            Paragraph("Tuned LightGBM classifier with early stopping. Strongest on TCS (41.6% accuracy).", style_table_cell_left),
        ],
        [
            Paragraph("<b>Interactive CLI Tool</b>", style_table_cell_left),
            Paragraph("<code>scripts/predict_sample.py</code>", style_table_cell_left),
            Paragraph("Run live predictions on any stock: <code>python scripts/predict_sample.py --symbol TCS.NS</code>", style_table_cell_left),
        ],
        [
            Paragraph("<b>Audit Reports</b>", style_table_cell_left),
            Paragraph("<code>models/horizon_5d/phase2_complete_metrics.json</code>", style_table_cell_left),
            Paragraph("Complete machine-readable metrics, confusion matrices, and financial simulations.", style_table_cell_left),
        ],
    ]
    deliv_table = Table(deliverables_data, colWidths=[110, 204, 190])
    deliv_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CARD_BG]),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(deliv_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # Section 8: Recommended Next Steps
    # ---------------------------------------------------------
    story.append(Paragraph("8. Actionable Recommendations for Phase 3", style_h1))
    story.append(Paragraph("<b>Three High-Impact Pathways to Increase Predictive Power:</b>", style_body_bold))
    story.append(Paragraph(
        "1. <b>Confidence Filtering (Trade Selectivity):</b> Instead of trading every session, require the model to have &gt;50% predicted probability before triggering an entry. This eliminates noisy low-conviction trades.",
        style_bullet
    ))
    story.append(Paragraph(
        "2. <b>Binary Target Formulation (Drop HOLD):</b> Reframe the prediction target to binary direction (`UP` vs `DOWN`) or large momentum breakouts (&gt; ±2.0%), eliminating the noisy middle zone.",
        style_bullet
    ))
    story.append(Paragraph(
        "3. <b>Multi-Horizon Models:</b> Expand this architecture to the <b>20-Day Horizon (`horizon_20d`)</b> where macroeconomic and trend signals typically have significantly higher signal-to-noise ratios than weekly fluctuations.",
        style_bullet
    ))
    story.append(Spacer(1, 14))

    # Sign-off box
    sign_off = Table([
        [
            Paragraph("<b>Prepared by:</b> Antigravity AI Assistant", style_callout),
            Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_callout),
            Paragraph("<b>Status:</b> Phase 1 & 2 Completed", style_callout),
        ]
    ], colWidths=[168, 168, 168])
    sign_off.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,0), (-1,-1), CARD_BG),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(sign_off)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
