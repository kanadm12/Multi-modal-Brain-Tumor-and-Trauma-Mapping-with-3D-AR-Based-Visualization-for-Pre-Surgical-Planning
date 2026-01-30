# =============================================================================
# CLINICAL REPORT GENERATOR
# 
# Generates comprehensive clinical reports from tumor segmentation
# Includes tumor volumes, locations, and clinical observations
# Exports to HTML and PDF formats
# =============================================================================

import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ reportlab not installed. PDF generation disabled. Install with: pip install reportlab")


# =============================================================================
# TUMOR CLASSIFICATION DETAILS
# =============================================================================

TUMOR_CLASSES = {
    "NCR": {
        "full_name": "Necrotic Core",
        "description": "Central necrotic/non-enhancing region of the tumor",
        "clinical_significance": "Indicates tumor cell death, often associated with aggressive tumor growth",
        "color": "#8B0000"
    },
    "ED": {
        "full_name": "Peritumoral Edema",
        "description": "Swelling around the tumor due to fluid accumulation",
        "clinical_significance": "May cause neurological symptoms due to pressure on surrounding brain tissue",
        "color": "#FFD700"
    },
    "ET": {
        "full_name": "Enhancing Tumor",
        "description": "Active, contrast-enhancing tumor tissue",
        "clinical_significance": "Represents viable, actively growing tumor cells with disrupted blood-brain barrier",
        "color": "#FF0000"
    }
}

BRAIN_REGIONS = {
    "frontal": {"z_range": (0.6, 1.0), "y_range": (0, 0.4)},
    "parietal": {"z_range": (0.6, 1.0), "y_range": (0.3, 0.7)},
    "occipital": {"z_range": (0.6, 1.0), "y_range": (0.6, 1.0)},
    "temporal": {"z_range": (0.3, 0.6), "y_range": (0.3, 0.7)},
    "cerebellum": {"z_range": (0, 0.3), "y_range": (0.5, 1.0)},
    "brainstem": {"z_range": (0.2, 0.4), "y_range": (0.7, 1.0)},
}


# =============================================================================
# REPORT GENERATOR CLASS
# =============================================================================

class ReportGenerator:
    """Generate clinical reports from tumor analysis"""
    
    def __init__(self, hospital_name: str = "Medical Imaging Center"):
        self.hospital_name = hospital_name
        self.styles = self._create_styles() if REPORTLAB_AVAILABLE else None
    
    def generate_report(self,
                        tumor_stats: Dict[str, Any],
                        patient_info: Optional[Dict[str, str]] = None,
                        doctor_info: Optional[Dict[str, str]] = None,
                        output_dir: Path = None,
                        session_id: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive tumor analysis report.
        
        Args:
            tumor_stats: Statistics from mesh generation
            patient_info: Patient details (name, age, id, etc.)
            doctor_info: Doctor credentials
            output_dir: Directory to save reports
            session_id: Unique session identifier
            
        Returns:
            Report data dictionary
        """
        output_dir = Path(output_dir) if output_dir else Path("./reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build report data
        report = self._build_report_data(tumor_stats, patient_info, doctor_info, session_id)
        
        # Generate HTML report
        html_path = output_dir / "report.html"
        self._generate_html_report(report, html_path)
        
        # Generate PDF report
        if REPORTLAB_AVAILABLE:
            pdf_path = output_dir / "report.pdf"
            self._generate_pdf_report(report, pdf_path)
            report["pdf_path"] = str(pdf_path)
        
        # Save JSON data
        json_path = output_dir / "report_data.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        report["html_path"] = str(html_path)
        report["json_path"] = str(json_path)
        
        return report
    
    def _build_report_data(self,
                           tumor_stats: Dict[str, Any],
                           patient_info: Optional[Dict[str, str]],
                           doctor_info: Optional[Dict[str, str]],
                           session_id: str) -> Dict[str, Any]:
        """Build structured report data"""
        
        # Calculate total tumor volume
        total_volume_cm3 = sum(
            stats.get("volume_cm3", 0) 
            for stats in tumor_stats.values()
        )
        
        # Calculate whole tumor (WT = NCR + ED + ET)
        wt_volume = total_volume_cm3
        
        # Calculate tumor core (TC = NCR + ET)
        tc_volume = (
            tumor_stats.get("NCR", {}).get("volume_cm3", 0) +
            tumor_stats.get("ET", {}).get("volume_cm3", 0)
        )
        
        # Calculate enhancing tumor
        et_volume = tumor_stats.get("ET", {}).get("volume_cm3", 0)
        
        # Determine tumor location
        location = self._estimate_tumor_location(tumor_stats)
        
        # Determine tumor grade estimate
        grade_estimate = self._estimate_tumor_grade(tumor_stats)
        
        # Build findings
        findings = self._generate_findings(tumor_stats, location, grade_estimate)
        
        # Build recommendations
        recommendations = self._generate_recommendations(tumor_stats, grade_estimate)
        
        return {
            "report_id": session_id or datetime.now().strftime("%Y%m%d%H%M%S"),
            "generated_at": datetime.now().isoformat(),
            "hospital_name": self.hospital_name,
            
            "patient": patient_info or {
                "name": "Not Provided",
                "id": "N/A",
                "age": "N/A",
                "gender": "N/A"
            },
            
            "doctor": doctor_info or {
                "name": "Attending Physician",
                "department": "Neuro-Oncology",
                "credentials": "M.D."
            },
            
            "tumor_analysis": {
                "classes": {
                    class_name: {
                        **TUMOR_CLASSES.get(class_name, {}),
                        **stats
                    }
                    for class_name, stats in tumor_stats.items()
                },
                "whole_tumor_volume_cm3": round(wt_volume, 3),
                "tumor_core_volume_cm3": round(tc_volume, 3),
                "enhancing_tumor_volume_cm3": round(et_volume, 3),
                "estimated_location": location,
                "estimated_grade": grade_estimate
            },
            
            "clinical_findings": findings,
            "recommendations": recommendations,
            
            "disclaimer": (
                "This report is generated by an AI-assisted analysis system and should be "
                "used as a supplementary tool only. All findings should be reviewed and "
                "confirmed by a qualified radiologist and treating physician. This automated "
                "analysis does not constitute a medical diagnosis."
            )
        }
    
    def _estimate_tumor_location(self, tumor_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate tumor location based on centroid"""
        
        # Get average centroid across all tumor classes
        centroids = []
        for stats in tumor_stats.values():
            if "centroid" in stats:
                centroids.append(stats["centroid"])
        
        if not centroids:
            return {"region": "Unknown", "hemisphere": "Unknown"}
        
        avg_centroid = np.mean(centroids, axis=0)
        
        # Normalize to 0-1 range (assuming standard BraTS dimensions)
        normalized = avg_centroid / np.array([155, 240, 240])  # D, H, W
        
        # Determine hemisphere
        if normalized[2] < 0.5:
            hemisphere = "Left"
        elif normalized[2] > 0.5:
            hemisphere = "Right"
        else:
            hemisphere = "Midline"
        
        # Determine region
        region = "Central"
        for region_name, bounds in BRAIN_REGIONS.items():
            z_min, z_max = bounds["z_range"]
            y_min, y_max = bounds["y_range"]
            
            if z_min <= normalized[0] <= z_max and y_min <= normalized[1] <= y_max:
                region = region_name.capitalize()
                break
        
        return {
            "region": region,
            "hemisphere": hemisphere,
            "coordinates": {
                "axial_slice": int(avg_centroid[0]),
                "coronal_slice": int(avg_centroid[1]),
                "sagittal_slice": int(avg_centroid[2])
            }
        }
    
    def _estimate_tumor_grade(self, tumor_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate tumor grade based on tumor characteristics"""
        
        et_volume = tumor_stats.get("ET", {}).get("volume_cm3", 0)
        ncr_volume = tumor_stats.get("NCR", {}).get("volume_cm3", 0)
        ed_volume = tumor_stats.get("ED", {}).get("volume_cm3", 0)
        total_volume = et_volume + ncr_volume + ed_volume
        
        if total_volume == 0:
            return {"grade": "Unable to determine", "confidence": "Low"}
        
        # Calculate ratios
        et_ratio = et_volume / total_volume if total_volume > 0 else 0
        ncr_ratio = ncr_volume / total_volume if total_volume > 0 else 0
        
        # Grade estimation based on WHO 2021 glioma classification principles
        if et_ratio > 0.3 and ncr_ratio > 0.1:
            grade = "High-Grade Glioma (Grade IV - likely GBM)"
            confidence = "Moderate"
            description = "Significant enhancing tumor with necrosis suggests aggressive pathology"
        elif et_ratio > 0.15:
            grade = "High-Grade Glioma (Grade III-IV)"
            confidence = "Moderate"
            description = "Presence of enhancement suggests malignant transformation"
        elif ed_volume > et_volume * 3:
            grade = "Low-Grade Glioma (Grade II)"
            confidence = "Low"
            description = "Predominantly edematous changes with minimal enhancement"
        else:
            grade = "Indeterminate"
            confidence = "Low"
            description = "Histopathological correlation required for accurate grading"
        
        return {
            "grade": grade,
            "confidence": confidence,
            "description": description,
            "enhancement_ratio": round(et_ratio * 100, 1),
            "necrosis_ratio": round(ncr_ratio * 100, 1)
        }
    
    def _generate_findings(self, 
                          tumor_stats: Dict[str, Any],
                          location: Dict[str, Any],
                          grade: Dict[str, Any]) -> List[str]:
        """Generate list of clinical findings"""
        
        findings = []
        
        # Location finding
        findings.append(
            f"A mass lesion is identified in the {location['region'].lower()} region "
            f"of the {location['hemisphere'].lower()} cerebral hemisphere."
        )
        
        # Volume findings
        for class_name, stats in tumor_stats.items():
            if stats.get("volume_cm3", 0) > 0:
                class_info = TUMOR_CLASSES.get(class_name, {})
                findings.append(
                    f"{class_info.get('full_name', class_name)}: {stats['volume_cm3']} cm³ "
                    f"({class_info.get('description', '')})"
                )
        
        # Grade finding
        if grade.get("grade"):
            findings.append(f"Imaging characteristics suggest: {grade['grade']}")
        
        # Additional observations
        if grade.get("enhancement_ratio", 0) > 30:
            findings.append(
                "Significant contrast enhancement noted, indicating active blood-brain barrier disruption."
            )
        
        if tumor_stats.get("NCR", {}).get("volume_cm3", 0) > 5:
            findings.append(
                "Central necrosis observed, which is characteristic of aggressive tumor behavior."
            )
        
        return findings
    
    def _generate_recommendations(self,
                                  tumor_stats: Dict[str, Any],
                                  grade: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations"""
        
        recommendations = [
            "Clinical correlation with patient symptoms recommended.",
            "Consider neurosurgical consultation for treatment planning.",
        ]
        
        total_volume = sum(s.get("volume_cm3", 0) for s in tumor_stats.values())
        
        if total_volume > 30:
            recommendations.append(
                "Large tumor volume noted. Urgent evaluation recommended."
            )
        
        if grade.get("confidence") == "Low":
            recommendations.append(
                "Histopathological confirmation through biopsy or resection recommended."
            )
        
        recommendations.extend([
            "Serial MRI imaging for treatment response monitoring.",
            "Multidisciplinary tumor board review recommended.",
        ])
        
        return recommendations
    
    def _create_styles(self):
        """Create PDF styles"""
        if not REPORTLAB_AVAILABLE:
            return None
        
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceBefore=20,
            spaceAfter=10
        ))
        
        styles.add(ParagraphStyle(
            name='BodyText',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=8
        ))
        
        return styles
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: Path):
        """Generate HTML report"""
        
        tumor_rows = ""
        for class_name, data in report["tumor_analysis"]["classes"].items():
            tumor_rows += f"""
                <tr>
                    <td><span class="color-box" style="background-color: {data.get('color', '#888')}"></span>
                        {data.get('full_name', class_name)}</td>
                    <td>{data.get('volume_cm3', 0)} cm³</td>
                    <td>{data.get('clinical_significance', 'N/A')}</td>
                </tr>
            """
        
        findings_html = "\n".join(f"<li>{f}</li>" for f in report["clinical_findings"])
        recommendations_html = "\n".join(f"<li>{r}</li>" for r in report["recommendations"])
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumor Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c5aa0;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header .hospital {{
            font-size: 18px;
            color: #666;
        }}
        .header .report-id {{
            font-size: 12px;
            color: #888;
            margin-top: 10px;
        }}
        .section {{
            margin-bottom: 25px;
        }}
        .section h2 {{
            color: #2c5aa0;
            font-size: 18px;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 1px solid #ddd;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .info-box {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
        }}
        .info-box label {{
            font-weight: bold;
            color: #555;
            display: block;
            margin-bottom: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #2c5aa0;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .color-box {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 10px;
            vertical-align: middle;
        }}
        .volume-summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .volume-card {{
            background: linear-gradient(135deg, #2c5aa0, #1e3d6e);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .volume-card .value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .volume-card .label {{
            font-size: 12px;
            opacity: 0.9;
        }}
        ul {{
            margin-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        .grade-box {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .disclaimer {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-top: 30px;
            font-size: 12px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #888;
            font-size: 12px;
        }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .container {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Brain Tumor Analysis Report</h1>
            <div class="hospital">{report['hospital_name']}</div>
            <div class="report-id">Report ID: {report['report_id']} | Generated: {report['generated_at']}</div>
        </div>
        
        <div class="section">
            <h2>Patient & Physician Information</h2>
            <div class="info-grid">
                <div class="info-box">
                    <label>Patient Name</label>
                    {report['patient']['name']}
                    <br><label>Patient ID</label>
                    {report['patient']['id']}
                    <br><label>Age / Gender</label>
                    {report['patient'].get('age', 'N/A')} / {report['patient'].get('gender', 'N/A')}
                </div>
                <div class="info-box">
                    <label>Physician</label>
                    {report['doctor']['name']}, {report['doctor'].get('credentials', '')}
                    <br><label>Department</label>
                    {report['doctor'].get('department', 'N/A')}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Tumor Volume Summary</h2>
            <div class="volume-summary">
                <div class="volume-card">
                    <div class="value">{report['tumor_analysis']['whole_tumor_volume_cm3']}</div>
                    <div class="label">Whole Tumor (cm³)</div>
                </div>
                <div class="volume-card">
                    <div class="value">{report['tumor_analysis']['tumor_core_volume_cm3']}</div>
                    <div class="label">Tumor Core (cm³)</div>
                </div>
                <div class="volume-card">
                    <div class="value">{report['tumor_analysis']['enhancing_tumor_volume_cm3']}</div>
                    <div class="label">Enhancing Tumor (cm³)</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Tumor Segmentation Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Tumor Region</th>
                        <th>Volume</th>
                        <th>Clinical Significance</th>
                    </tr>
                </thead>
                <tbody>
                    {tumor_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Tumor Location</h2>
            <p><strong>Region:</strong> {report['tumor_analysis']['estimated_location']['region']}</p>
            <p><strong>Hemisphere:</strong> {report['tumor_analysis']['estimated_location']['hemisphere']}</p>
        </div>
        
        <div class="section">
            <h2>Grade Assessment</h2>
            <div class="grade-box">
                <strong>{report['tumor_analysis']['estimated_grade']['grade']}</strong><br>
                <small>Confidence: {report['tumor_analysis']['estimated_grade']['confidence']}</small><br>
                {report['tumor_analysis']['estimated_grade'].get('description', '')}
            </div>
        </div>
        
        <div class="section">
            <h2>Clinical Findings</h2>
            <ul>
                {findings_html}
            </ul>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
                {recommendations_html}
            </ul>
        </div>
        
        <div class="disclaimer">
            <strong>⚠️ Disclaimer:</strong> {report['disclaimer']}
        </div>
        
        <div class="footer">
            <p>Generated by AI-Assisted Brain Tumor Analysis System</p>
            <p>For clinical use, please verify all findings with qualified medical professionals</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _generate_pdf_report(self, report: Dict[str, Any], output_path: Path):
        """Generate PDF report using ReportLab"""
        
        if not REPORTLAB_AVAILABLE:
            return
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30
        )
        
        elements = []
        
        # Title
        elements.append(Paragraph(
            "Brain Tumor Analysis Report",
            self.styles['ReportTitle']
        ))
        elements.append(Paragraph(
            f"<i>{report['hospital_name']}</i>",
            ParagraphStyle('Center', alignment=TA_CENTER)
        ))
        elements.append(Spacer(1, 20))
        
        # Patient Info Table
        elements.append(Paragraph("Patient Information", self.styles['SectionHeader']))
        patient_data = [
            ["Patient Name:", report['patient']['name'], "Patient ID:", report['patient']['id']],
            ["Age:", report['patient'].get('age', 'N/A'), "Gender:", report['patient'].get('gender', 'N/A')],
            ["Physician:", report['doctor']['name'], "Department:", report['doctor'].get('department', 'N/A')],
        ]
        patient_table = Table(patient_data, colWidths=[100, 150, 100, 150])
        patient_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 20))
        
        # Volume Summary
        elements.append(Paragraph("Tumor Volume Summary", self.styles['SectionHeader']))
        volume_data = [
            ["Whole Tumor", "Tumor Core", "Enhancing Tumor"],
            [
                f"{report['tumor_analysis']['whole_tumor_volume_cm3']} cm³",
                f"{report['tumor_analysis']['tumor_core_volume_cm3']} cm³",
                f"{report['tumor_analysis']['enhancing_tumor_volume_cm3']} cm³"
            ]
        ]
        volume_table = Table(volume_data, colWidths=[170, 170, 170])
        volume_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(volume_table)
        elements.append(Spacer(1, 20))
        
        # Tumor Details
        elements.append(Paragraph("Segmentation Details", self.styles['SectionHeader']))
        tumor_header = ["Region", "Volume (cm³)", "Clinical Significance"]
        tumor_data = [tumor_header]
        for class_name, data in report["tumor_analysis"]["classes"].items():
            tumor_data.append([
                data.get('full_name', class_name),
                str(data.get('volume_cm3', 0)),
                data.get('clinical_significance', 'N/A')[:60] + "..."
            ])
        
        tumor_table = Table(tumor_data, colWidths=[120, 80, 310])
        tumor_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(tumor_table)
        elements.append(Spacer(1, 20))
        
        # Location
        elements.append(Paragraph("Tumor Location", self.styles['SectionHeader']))
        loc = report['tumor_analysis']['estimated_location']
        elements.append(Paragraph(
            f"<b>Region:</b> {loc['region']} | <b>Hemisphere:</b> {loc['hemisphere']}",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 10))
        
        # Grade
        elements.append(Paragraph("Grade Assessment", self.styles['SectionHeader']))
        grade = report['tumor_analysis']['estimated_grade']
        elements.append(Paragraph(
            f"<b>{grade['grade']}</b><br/>"
            f"Confidence: {grade['confidence']}<br/>"
            f"{grade.get('description', '')}",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 15))
        
        # Findings
        elements.append(Paragraph("Clinical Findings", self.styles['SectionHeader']))
        for finding in report["clinical_findings"]:
            elements.append(Paragraph(f"• {finding}", self.styles['BodyText']))
        elements.append(Spacer(1, 15))
        
        # Recommendations
        elements.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        for rec in report["recommendations"]:
            elements.append(Paragraph(f"• {rec}", self.styles['BodyText']))
        elements.append(Spacer(1, 20))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.red,
            borderColor=colors.red,
            borderWidth=1,
            borderPadding=10
        )
        elements.append(Paragraph(
            f"<b>Disclaimer:</b> {report['disclaimer']}",
            disclaimer_style
        ))
        
        # Build PDF
        doc.build(elements)


# =============================================================================
# CLI USAGE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate clinical report")
    parser.add_argument("--stats", type=str, required=True, help="Path to tumor stats JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--patient-name", type=str, default="Test Patient")
    parser.add_argument("--patient-id", type=str, default="PAT001")
    
    args = parser.parse_args()
    
    with open(args.stats) as f:
        tumor_stats = json.load(f)
    
    generator = ReportGenerator()
    report = generator.generate_report(
        tumor_stats=tumor_stats.get("tumor_stats", tumor_stats),
        patient_info={"name": args.patient_name, "id": args.patient_id},
        output_dir=Path(args.output)
    )
    
    print(f"✅ Report generated: {report.get('html_path')}")
