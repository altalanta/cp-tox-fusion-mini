"""Report generation and documentation compilation."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import markdown
import pandas as pd


def load_json_file(file_path: Path) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing {file_path}: {e}")
        return {}


def format_risk_assessment(leakage_data: Dict) -> str:
    """Format leakage risk assessment for markdown."""
    if not leakage_data:
        return "**Status**: No leakage assessment available"
    
    risk_info = leakage_data.get("leakage_risk", {})
    overall_risk = risk_info.get("overall_risk", "unknown")
    risk_factors = risk_info.get("risk_factors", [])
    
    # Color coding for risk levels
    risk_colors = {
        "low": "🟢",
        "medium": "🟡", 
        "high": "🔴",
        "unknown": "⚪"
    }
    
    risk_emoji = risk_colors.get(overall_risk, "⚪")
    
    markdown_text = f"**Overall Risk Level**: {risk_emoji} {overall_risk.upper()}\n\n"
    
    if risk_factors:
        markdown_text += "**Risk Factors Detected**:\n"
        for factor in risk_factors:
            markdown_text += f"- {factor}\n"
    else:
        markdown_text += "**Risk Factors**: None detected\n"
    
    # Add detailed probe results
    plate_layout = leakage_data.get("plate_layout_probe", {})
    plate_effect = leakage_data.get("plate_effect_on_target", {})
    permutation = leakage_data.get("permutation_test", {})
    
    markdown_text += "\n**Diagnostic Details**:\n"
    
    if "max_score" in plate_layout:
        markdown_text += f"- Maximum Probe Score: {plate_layout['max_score']:.3f}\n"
    
    if "plate_effect_score" in plate_effect:
        markdown_text += f"- Plate Effect on Target: {plate_effect['plate_effect_score']:.3f}\n"
    
    if "p_value" in permutation:
        markdown_text += f"- Permutation Test p-value: {permutation['p_value']:.3f}\n"
    
    return markdown_text


def format_metrics_table(metrics_data: Dict) -> str:
    """Format model metrics as markdown table."""
    if not metrics_data:
        return "| Metric | Value |\n|--------|-------|\n| Status | No metrics available |"
    
    # Key metrics to display
    key_metrics = [
        ("AUROC", "auroc"),
        ("Average Precision", "ap"), 
        ("Accuracy", "accuracy"),
        ("Brier Score", "brier"),
        ("Expected Calibration Error", "ece"),
        ("Training Samples", "n_train"),
        ("Test Samples", "n_test"),
        ("Features", "n_features")
    ]
    
    table_rows = ["| Metric | Value |", "|--------|-------|"]
    
    for display_name, key in key_metrics:
        if key in metrics_data:
            value = metrics_data[key]
            if isinstance(value, float):
                if 0 <= value <= 1:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            table_rows.append(f"| {display_name} | {formatted_value} |")
    
    return "\n".join(table_rows)


def format_ic50_summary(ic50_data: Dict) -> str:
    """Format IC50 summary for markdown."""
    if not ic50_data:
        return "**Status**: No IC50 analysis available"
    
    summary_stats = ic50_data.get("summary_statistics", {})
    
    markdown_text = "**IC50 Analysis Summary**:\n"
    markdown_text += f"- Total Compounds: {summary_stats.get('total_compounds', 0)}\n"
    markdown_text += f"- Successful Fits: {summary_stats.get('successful_fits', 0)}\n"
    markdown_text += f"- Failed Fits: {summary_stats.get('failed_fits', 0)}\n"
    markdown_text += f"- Extrapolated IC50s: {summary_stats.get('extrapolated_fits', 0)}\n"
    
    # Sample results
    compound_results = ic50_data.get("compound_results", {})
    if compound_results:
        markdown_text += "\n**Sample IC50 Values**:\n"
        sample_compounds = list(compound_results.keys())[:5]  # First 5 compounds
        
        for compound_id in sample_compounds:
            result = compound_results[compound_id]
            if result.get("fit_success", False):
                ic50_val = result.get("ic50", "N/A")
                r_squared = result.get("r_squared", "N/A")
                if isinstance(ic50_val, (int, float)) and isinstance(r_squared, (int, float)):
                    markdown_text += f"- {compound_id}: IC50 = {ic50_val:.2e}, R² = {r_squared:.3f}\n"
                else:
                    markdown_text += f"- {compound_id}: Fit failed\n"
    
    return markdown_text


def embed_figures(figures_dir: Path, base_path: str = "figures") -> str:
    """Generate markdown for embedding figures."""
    if not figures_dir.exists():
        return ""
    
    markdown_text = ""
    
    # Main evaluation figures
    figure_files = [
        ("roc.png", "ROC Curve", "Receiver Operating Characteristic curve showing model discrimination performance."),
        ("pr.png", "Precision-Recall Curve", "Precision-Recall curve showing performance across different thresholds."),
        ("calibration.png", "Calibration Plot", "Calibration curve showing how well predicted probabilities match actual outcomes.")
    ]
    
    for filename, title, description in figure_files:
        figure_path = figures_dir / filename
        if figure_path.exists():
            relative_path = f"{base_path}/{filename}"
            markdown_text += f"\n### {title}\n\n"
            markdown_text += f"{description}\n\n"
            markdown_text += f"![{title}]({relative_path})\n\n"
    
    # Leakage probe figures
    leakage_figures = [
        ("leakage_probe_plate.png", "Plate Confounding Probe", "Results of plate ID predictability from features."),
        ("leakage_probe_layout.png", "Layout Confounding Probe", "Results of well position predictability from features.")
    ]
    
    for filename, title, description in leakage_figures:
        figure_path = figures_dir / filename
        if figure_path.exists():
            relative_path = f"{base_path}/{filename}"
            markdown_text += f"\n### {title}\n\n"
            markdown_text += f"{description}\n\n"
            markdown_text += f"![{title}]({relative_path})\n\n"
    
    return markdown_text


def generate_main_report(reports_dir: Path = Path("reports"),
                        model_card_path: Path = Path("model_card.md")) -> str:
    """Generate main markdown report combining all results."""
    
    # Load data files
    metrics_data = load_json_file(reports_dir / "model_metrics.json")
    leakage_data = load_json_file(reports_dir / "leakage.json")
    ic50_data = load_json_file(reports_dir / "ic50_summary.json")
    
    # Load model card if available
    model_card_content = ""
    if model_card_path.exists():
        with open(model_card_path, 'r') as f:
            model_card_content = f.read()
    
    # Generate report
    report_lines = [
        "# CP-Tox-Mini Analysis Report",
        "",
        f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "## Executive Summary",
        "",
        "This report presents the results of a Cell Painting × toxicity fusion analysis using the CP-Tox-Mini pipeline. ",
        "The analysis includes model performance metrics, data quality assessment, leakage detection, and dose-response modeling.",
        "",
        "## Model Performance",
        "",
        format_metrics_table(metrics_data),
        "",
        "## Risk of Leakage Assessment", 
        "",
        format_risk_assessment(leakage_data),
        "",
        "## IC50 Dose-Response Analysis",
        "",
        format_ic50_summary(ic50_data),
        "",
        "## Visualizations",
        "",
        embed_figures(reports_dir / "figures"),
        "",
        "## Technical Details",
        "",
        "### Data Processing",
        "- **Deterministic Processing**: All operations use fixed random seeds (42) for reproducibility",
        "- **Data Validation**: Input files validated against SHA256 manifest checksums", 
        "- **Quality Control**: Automated checks for image quality, feature completeness, and data integrity",
        "",
        "### Model Architecture",
        "- **Baseline Model**: Logistic Regression with StandardScaler normalization",
        "- **Features**: Morphological features from Cell Painting + chemical descriptors from RDKit",
        "- **Evaluation**: 70/30 train/test split with stratification",
        "",
        "### Limitations",
        "- **Dataset Size**: Small training set suitable for smoke testing only",
        "- **Generalizability**: Limited to demonstration purposes, not validated for production use",
        "- **Batch Effects**: Potential confounding detected - see Risk Assessment above",
        "",
        "---",
        "",
        "*This report was generated automatically by the CP-Tox-Mini pipeline.*"
    ]
    
    return "\n".join(report_lines)


def render_markdown_to_html(markdown_content: str, title: str = "CP-Tox-Mini Report") -> str:
    """Convert markdown content to HTML with CSS styling."""
    
    # Convert markdown to HTML
    html_body = markdown.markdown(markdown_content, extensions=['tables', 'toc'])
    
    # CSS styling
    css_styles = """
    <style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
        background-color: #fff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        margin-top: 2em;
        margin-bottom: 1em;
    }
    h1 { font-size: 2.5em; border-bottom: 3px solid #3498db; padding-bottom: 0.3em; }
    h2 { font-size: 2em; border-bottom: 2px solid #ecf0f1; padding-bottom: 0.3em; }
    h3 { font-size: 1.5em; color: #34495e; }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    th {
        background-color: #f8f9fa;
        font-weight: 600;
        color: #2c3e50;
    }
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    tr:hover {
        background-color: #e8f4f8;
    }
    img {
        max-width: 100%;
        height: auto;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1em 0;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Monaco', 'Consolas', monospace;
    }
    pre {
        background-color: #f8f8f8;
        padding: 1em;
        border-radius: 5px;
        overflow-x: auto;
    }
    blockquote {
        border-left: 4px solid #3498db;
        margin: 1em 0;
        padding: 0.5em 1em;
        background-color: #f8f9fa;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    hr {
        border: none;
        border-top: 2px solid #ecf0f1;
        margin: 2em 0;
    }
    .footer {
        margin-top: 3em;
        padding-top: 1em;
        border-top: 1px solid #ecf0f1;
        font-size: 0.9em;
        color: #7f8c8d;
    }
    </style>
    """
    
    # Complete HTML document
    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {css_styles}
</head>
<body>
    {html_body}
    <div class="footer">
        <p>Generated by CP-Tox-Mini Pipeline • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
</body>
</html>"""
    
    return html_document


def create_index_page(reports_dir: Path = Path("reports")) -> str:
    """Create landing page HTML that links to all reports and metrics."""
    
    # Load summary data
    metrics_data = load_json_file(reports_dir / "model_metrics.json")
    leakage_data = load_json_file(reports_dir / "leakage.json")
    ic50_data = load_json_file(reports_dir / "ic50_summary.json")
    
    # Extract key metrics for summary table
    auroc = metrics_data.get("auroc", "N/A")
    ap = metrics_data.get("ap", "N/A")
    accuracy = metrics_data.get("accuracy", "N/A")
    
    risk_level = leakage_data.get("leakage_risk", {}).get("overall_risk", "unknown")
    
    ic50_compounds = ic50_data.get("summary_statistics", {}).get("total_compounds", 0)
    ic50_successful = ic50_data.get("summary_statistics", {}).get("successful_fits", 0)
    
    # Format metrics for display
    def format_metric(value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CP-Tox-Mini Dashboard</title>
    <style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }}
    .container {{
        background: white;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
    h1 {{
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5em;
        font-size: 2.5em;
    }}
    .subtitle {{
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2em;
        font-size: 1.2em;
    }}
    .dashboard-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 2em 0;
    }}
    .metric-card {{
        background: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        border-left: 4px solid #3498db;
    }}
    .metric-card h3 {{
        margin-top: 0;
        color: #2c3e50;
    }}
    .metric-value {{
        font-size: 2em;
        font-weight: bold;
        color: #3498db;
    }}
    .risk-high {{
        color: #e74c3c;
        border-left-color: #e74c3c;
    }}
    .risk-medium {{
        color: #f39c12;
        border-left-color: #f39c12;
    }}
    .risk-low {{
        color: #27ae60;
        border-left-color: #27ae60;
    }}
    .links-section {{
        margin: 2em 0;
    }}
    .link-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
    }}
    .link-card {{
        background: white;
        border: 2px solid #ecf0f1;
        border-radius: 8px;
        padding: 20px;
        text-decoration: none;
        color: #2c3e50;
        transition: all 0.3s ease;
    }}
    .link-card:hover {{
        border-color: #3498db;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
    }}
    .link-card h4 {{
        margin: 0 0 10px 0;
        color: #3498db;
    }}
    .timestamp {{
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9em;
        margin-top: 2em;
        padding-top: 1em;
        border-top: 1px solid #ecf0f1;
    }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 CP-Tox-Mini Dashboard</h1>
        <p class="subtitle">Cell Painting × Toxicity Fusion Analysis Results</p>
        
        <div class="dashboard-grid">
            <div class="metric-card">
                <h3>Model Performance</h3>
                <div class="metric-value">{format_metric(auroc)}</div>
                <p>AUROC Score</p>
                <small>Average Precision: {format_metric(ap)} | Accuracy: {format_metric(accuracy)}</small>
            </div>
            
            <div class="metric-card risk-{risk_level}">
                <h3>Leakage Risk</h3>
                <div class="metric-value">{risk_level.upper()}</div>
                <p>Overall Risk Level</p>
                <small>Data quality and batch effect assessment</small>
            </div>
            
            <div class="metric-card">
                <h3>IC50 Analysis</h3>
                <div class="metric-value">{ic50_successful}/{ic50_compounds}</div>
                <p>Successful Fits</p>
                <small>Dose-response curve fitting results</small>
            </div>
        </div>
        
        <div class="links-section">
            <h2>📊 Reports and Documentation</h2>
            <div class="link-grid">
                <a href="cp-tox-mini_report.html" class="link-card">
                    <h4>📋 Main Report</h4>
                    <p>Complete analysis report with metrics, visualizations, and technical details</p>
                </a>
                
                <a href="../model_card.md" class="link-card">
                    <h4>🏷️ Model Card</h4>
                    <p>Model documentation, limitations, and ethical considerations</p>
                </a>
                
                <a href="model_metrics.json" class="link-card">
                    <h4>📈 Metrics (JSON)</h4>
                    <p>Raw model performance metrics and evaluation results</p>
                </a>
                
                <a href="leakage.json" class="link-card">
                    <h4>🔍 Leakage Assessment</h4>
                    <p>Data quality diagnostics and batch effect analysis</p>
                </a>
                
                <a href="ic50_summary.json" class="link-card">
                    <h4>💊 IC50 Results</h4>
                    <p>Dose-response modeling and IC50 estimation results</p>
                </a>
                
                <a href="figures/" class="link-card">
                    <h4>📸 Figures</h4>
                    <p>ROC curves, calibration plots, and diagnostic visualizations</p>
                </a>
            </div>
        </div>
        
        <div class="timestamp">
            <p>Generated on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M:%S')} UTC by CP-Tox-Mini Pipeline</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_content


def build_reports(reports_dir: Path = Path("reports"),
                 docs_dir: Path = Path("docs"),
                 model_card_path: Path = Path("model_card.md")) -> bool:
    """Build all reports and documentation."""
    
    print("Building reports and documentation...")
    
    # Create directories
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate main markdown report
    print("  Generating main report...")
    markdown_report = generate_main_report(reports_dir, model_card_path)
    
    # Save markdown report
    markdown_path = reports_dir / "cp-tox-mini_report.md"
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)
    
    # Convert to HTML
    html_report = render_markdown_to_html(markdown_report)
    html_path = reports_dir / "cp-tox-mini_report.html"
    with open(html_path, 'w') as f:
        f.write(html_report)
    
    # Create index/dashboard page
    print("  Creating index page...")
    index_html = create_index_page(reports_dir)
    index_path = reports_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    # Copy reports to docs for GitHub Pages
    print("  Copying to docs directory for GitHub Pages...")
    
    # Copy HTML files
    if (reports_dir / "index.html").exists():
        shutil.copy2(reports_dir / "index.html", docs_dir / "index.html")
    
    if (reports_dir / "cp-tox-mini_report.html").exists():
        shutil.copy2(reports_dir / "cp-tox-mini_report.html", docs_dir / "cp-tox-mini_report.html")
    
    # Copy figures directory
    figures_src = reports_dir / "figures"
    figures_dst = docs_dir / "figures"
    if figures_src.exists():
        if figures_dst.exists():
            shutil.rmtree(figures_dst)
        shutil.copytree(figures_src, figures_dst)
    
    # Copy JSON files
    for json_file in ["model_metrics.json", "leakage.json", "ic50_summary.json"]:
        src_path = reports_dir / json_file
        if src_path.exists():
            shutil.copy2(src_path, docs_dir / json_file)
    
    # Copy model card
    if model_card_path.exists():
        shutil.copy2(model_card_path, docs_dir / "model_card.md")
    
    print(f"  Reports saved to {reports_dir}")
    print(f"  Documentation copied to {docs_dir}")
    print(f"  Main report: {html_path}")
    print(f"  Dashboard: {index_path}")
    
    return True


if __name__ == "__main__":
    # CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reporting.py <command> [args...]")
        print("Commands:")
        print("  build [reports_dir] [docs_dir] - Build all reports")
        print("  markdown [reports_dir]         - Generate markdown report only")
        print("  index [reports_dir]            - Generate index page only")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "build":
        reports_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports")
        docs_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("docs")
        
        success = build_reports(reports_dir, docs_dir)
        sys.exit(0 if success else 1)
        
    elif command == "markdown":
        reports_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports")
        
        markdown_content = generate_main_report(reports_dir)
        print(markdown_content)
        
    elif command == "index":
        reports_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports")
        
        index_content = create_index_page(reports_dir)
        print(index_content)
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)