"""
Report generator module.
Generates professional PDF reports using ReportLab.
"""
import io
import os
from datetime import datetime


def generate_pdf_report(results):
    """
    Generate a professional PDF report from analysis results.
    
    results dict should contain:
        - crop_name
        - area_ha
        - n_plants
        - planting_density
        - mean_slope_deg
        - surface_area_ha
        - avg_eto
        - avg_etc
        - total_irrigation_mm
        - irrigation_days
        - revenue
        - cost
        - profit
        - profit_margin_pct
        - charts: list of (title, PIL Image) tuples
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        spaceAfter=6,
        textColor=colors.HexColor("#1B5E20"),
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=20,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor("#2E7D32"),
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    label_style = ParagraphStyle(
        "LabelStyle",
        parent=body_style,
        textColor=colors.HexColor("#666666"),
        fontSize=9,
    )

    elements = []

    # Title page content
    elements.append(Spacer(1, 40 * mm))
    elements.append(Paragraph("PRECISION AGRICULTURE", title_style))
    elements.append(Paragraph("OPTIMIZATION FRAMEWORK", title_style))
    elements.append(Spacer(1, 10 * mm))
    elements.append(HRFlowable(
        width="60%", thickness=2, color=colors.HexColor("#2E7D32"),
        spaceAfter=12, spaceBefore=0, hAlign="CENTER",
    ))
    elements.append(Spacer(1, 5 * mm))
    elements.append(Paragraph(
        f"Crop Suitability & Optimization Analysis Report",
        subtitle_style,
    ))
    elements.append(Spacer(1, 15 * mm))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        ParagraphStyle("DateStyle", parent=body_style, alignment=TA_CENTER, textColor=colors.HexColor("#888888")),
    ))
    elements.append(Paragraph(
        f"Location: Uganda (Lat {results.get('lat', 0.35)}, Lon {results.get('lon', 33.75)})",
        ParagraphStyle("LocStyle", parent=body_style, alignment=TA_CENTER, textColor=colors.HexColor("#888888")),
    ))
    elements.append(PageBreak())

    # 1. Selected Crop
    elements.append(Paragraph("1. CROP SELECTION", heading_style))
    crop_table = Table(
        [
            ["Parameter", "Value"],
            ["Selected Crop", results.get("crop_name", "N/A")],
            ["Farm Area (ha)", f"{results.get('area_ha', 0):.2f}"],
            ["Number of Plants", f"{results.get('n_plants', 0):,}"],
            ["Plant Density (plants/ha)", f"{results.get('planting_density', 0):.0f}"],
        ],
        colWidths=[140 * mm, 60 * mm],
    )
    crop_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(crop_table)

    # 2. Topography Results
    elements.append(Paragraph("2. TOPOGRAPHY ANALYSIS", heading_style))
    topo_table = Table(
        [
            ["Parameter", "Value"],
            ["Mean Slope (degrees)", f"{results.get('mean_slope_deg', 0):.2f}"],
            ["Maximum Slope (degrees)", f"{results.get('max_slope_deg', 0):.2f}"],
            ["Surface Area (ha)", f"{results.get('surface_area_ha', 0):.2f}"],
        ],
        colWidths=[140 * mm, 60 * mm],
    )
    topo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(topo_table)

    # 3. Weather Summary
    elements.append(Paragraph("3. WEATHER SUMMARY", heading_style))
    weather = results.get("weather_summary", {})
    weather_table = Table(
        [
            ["Parameter", "Value"],
            ["Avg Max Temperature", f"{weather.get('avg_tmax', 0):.1f} °C"],
            ["Avg Min Temperature", f"{weather.get('avg_tmin', 0):.1f} °C"],
            ["Total Rainfall", f"{weather.get('total_rainfall', 0):.1f} mm"],
            ["Avg Solar Radiation", f"{weather.get('avg_rs', 0):.1f} MJ/m²/day"],
        ],
        colWidths=[140 * mm, 60 * mm],
    )
    weather_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(weather_table)

    # 4. Water Requirements
    elements.append(Paragraph("4. WATER REQUIREMENT ANALYSIS", heading_style))
    water_table = Table(
        [
            ["Parameter", "Value"],
            ["Average ET0 (mm/day)", f"{results.get('avg_eto', 0):.2f}"],
            ["Average ETc (mm/day)", f"{results.get('avg_etc', 0):.2f}"],
            ["Total Irrigation Demand (mm)", f"{results.get('total_irrigation_mm', 0):.1f}"],
            ["Irrigation Required Days", f"{results.get('irrigation_days', 0)}"],
        ],
        colWidths=[140 * mm, 60 * mm],
    )
    water_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(water_table)

    # 5. Yield Prediction
    elements.append(Paragraph("5. YIELD PREDICTION", heading_style))
    yield_pred_model = results.get("yield_prediction_best_model", "N/A")
    yield_pred_rmse = results.get("best_rmse", "N/A")
    expected_yield = results.get("expected_yield", "N/A")
    yp_table = Table(
        [
            ["Parameter", "Value"],
            ["Best Performing Model", str(yield_pred_model)],
            ["Best Test RMSE", str(yield_pred_rmse)],
            ["Expected Yield (t/ha)", str(expected_yield) if expected_yield else "N/A"],
        ],
        colWidths=[140 * mm, 60 * mm],
    )
    yp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(yp_table)

    # 6. Economic Analysis
    elements.append(Paragraph("6. ECONOMIC ANALYSIS", heading_style))
    revenue_val = results.get('revenue', 0)
    cost_val = results.get('cost', 0)
    profit_val_econ = results.get('profit', 0)
    margin_val = results.get('profit_margin_pct', 0)
    wue_val = results.get("wue", "N/A")
    iwue_val = results.get("iwue", "N/A")
    econ_table = Table(
        [
            ["Parameter", "Value"],
            ["Total Revenue (UGX)", f"{revenue_val:,.0f}"],
            ["Total Production Cost (UGX)", f"{cost_val:,.0f}"],
            ["Net Profit (UGX)", f"{profit_val_econ:,.0f}"],
            ["Profit Margin", f"{margin_val:.1f}%"],
            ["WUE (kg/m³)", str(wue_val)],
            ["IWUE (kg/m³)", str(iwue_val)],
        ],
        colWidths=[140 * mm, 60 * mm],
    )
    econ_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(econ_table)

    # Charts
    chart_idx = 7
    for chart_title, pil_image in results.get("charts", []):
        elements.append(Spacer(1, 10 * mm))
        elements.append(Paragraph(f"{chart_idx}. {chart_title}", heading_style))
        chart_idx += 1
        img_path = f"/tmp/{chart_title.replace(' ', '_').lower()}.png"
        pil_image.save(img_path)
        elements.append(Image(img_path, width=160 * mm, height=90 * mm))

    # Conclusion
    elements.append(PageBreak())
    elements.append(Paragraph("CONCLUSION", heading_style))
    profit_val = results.get("profit", 0)
    if profit_val > 0:
        conclusion = (
            f"The Precision Agriculture Decision Support System analysis indicates that "
            f"cultivation of <b>{results.get('crop_name', 'the selected crop')}</b> on a "
            f"{results.get('area_ha', 0):.2f} ha farm is <b>economically viable</b>, "
            f"with an estimated profit of <b>{profit_val:,.0f} UGX</b> "
            f"({results.get('profit_margin_pct', 0):.1f}% margin). "
            f"The crop water requirement analysis shows an average ETc of "
            f"{results.get('avg_etc', 0):.2f} mm/day, with irrigation needed on "
            f"{results.get('irrigation_days', 0)} days during the growing season. "
            f"Yield prediction using <b>{results.get('yield_prediction_best_model', 'N/A')}</b> "
            f"achieved a test RMSE of {results.get('best_rmse', 'N/A')}, "
            f"with expected yield of {results.get('expected_yield', 'N/A')} t/ha."
        )
    else:
        conclusion = (
            f"The analysis indicates that cultivation may not be economically viable "
            f"under current parameters. Consider reviewing crop selection, input costs, "
            f"or exploring value-added processing opportunities."
        )
    elements.append(Paragraph(conclusion, body_style))
    elements.append(Spacer(1, 15 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    elements.append(Spacer(1, 5 * mm))
    elements.append(Paragraph(
        f"Report generated by Precision Agriculture Optimization Framework | "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.HexColor("#999999"), alignment=TA_CENTER),
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
