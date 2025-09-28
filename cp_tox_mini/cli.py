"""Command-line interface for CP-Tox-Mini pipeline."""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
import pandas as pd

from . import io, hashing, features, eval, diagnostics, dose_response, reporting

app = typer.Typer(
    name="cp-tox-mini",
    help="Cell Painting × Toxicity Fusion Pipeline",
    add_completion=False
)


def setup_environment():
    """Set up deterministic environment variables."""
    os.environ["PYTHONHASHSEED"] = "0"
    
    # Set NumPy/scikit-learn to single-threaded for reproducibility
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


@app.command()
def download(
    data_dir: Path = typer.Option(Path("data"), help="Data directory"),
    validate: bool = typer.Option(True, help="Validate downloads against manifest")
):
    """Download datasets and generate manifest."""
    setup_environment()
    
    typer.echo("🔽 Downloading datasets...")
    
    success = io.download_inputs(data_dir)
    
    if success:
        typer.echo("✅ Download completed successfully")
        
        if validate:
            typer.echo("🔍 Validating downloads...")
            if io.validate_inputs():
                typer.echo("✅ Validation passed")
            else:
                typer.echo("❌ Validation failed")
                raise typer.Exit(1)
    else:
        typer.echo("❌ Download failed")
        raise typer.Exit(1)


@app.command()
def features(
    data_dir: Path = typer.Option(Path("data"), help="Data directory"),
    output_path: Path = typer.Option(Path("data/processed/features.parquet"), help="Output features file"),
    n_samples: int = typer.Option(100, help="Number of synthetic samples to generate"),
    force: bool = typer.Option(False, help="Force regeneration if output exists")
):
    """Compute Cell Painting and chemical features."""
    setup_environment()
    
    typer.echo("🧪 Computing features...")
    
    if output_path.exists() and not force:
        typer.echo(f"⚠️  Features already exist at {output_path}, use --force to regenerate")
        return
    
    # For demo, generate synthetic features
    typer.echo(f"Generating {n_samples} synthetic samples...")
    feature_data = features.create_synthetic_features(n_samples=n_samples, random_state=42)
    
    # Process features
    processed_data, processing_info = features.process_features(feature_data)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_parquet(output_path, index=False)
    
    typer.echo(f"✅ Features computed: {processing_info['final_samples']} samples, {processing_info['final_features']} features")
    typer.echo(f"📁 Saved to {output_path}")


@app.command()
def fuse(
    features_path: Path = typer.Option(Path("data/processed/features.parquet"), help="Features file"),
    output_path: Path = typer.Option(Path("data/processed/fused.parquet"), help="Output fused file"),
    force: bool = typer.Option(False, help="Force regeneration if output exists")
):
    """Join modalities and create train/test splits."""
    setup_environment()
    
    typer.echo("🔗 Fusing modalities...")
    
    if output_path.exists() and not force:
        typer.echo(f"⚠️  Fused data already exists at {output_path}, use --force to regenerate")
        return
    
    if not features_path.exists():
        typer.echo(f"❌ Features file not found: {features_path}")
        typer.echo("Run 'cp-tox-mini features' first")
        raise typer.Exit(1)
    
    # Load features
    data = pd.read_parquet(features_path)
    
    # For demo, the data is already "fused" since we generated synthetic multimodal data
    # In a real pipeline, this would join CP and chemical features
    
    typer.echo(f"✅ Loaded {len(data)} samples")
    
    # Save fused data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False)
    
    typer.echo(f"✅ Fusion completed: {len(data)} samples")
    typer.echo(f"📁 Saved to {output_path}")


@app.command()
def train(
    data_path: Path = typer.Option(Path("data/processed/fused.parquet"), help="Fused data file"),
    model_type: str = typer.Option("logistic", help="Model type (logistic, rf)"),
    target_col: str = typer.Option("target", help="Target column name"),
    force: bool = typer.Option(False, help="Force retraining if model exists")
):
    """Train baseline model."""
    setup_environment()
    
    typer.echo(f"🤖 Training {model_type} model...")
    
    if not data_path.exists():
        typer.echo(f"❌ Data file not found: {data_path}")
        typer.echo("Run 'cp-tox-mini fuse' first")
        raise typer.Exit(1)
    
    # Load data
    data = pd.read_parquet(data_path)
    
    # Check for required columns
    if target_col not in data.columns:
        typer.echo(f"❌ Target column '{target_col}' not found")
        raise typer.Exit(1)
    
    # Train and evaluate
    metrics = eval.evaluate_model_and_save(
        data, 
        target_col=target_col,
        model_type=model_type,
        random_state=42
    )
    
    typer.echo(f"✅ Training completed")
    typer.echo(f"📊 Test AUROC: {metrics['auroc']:.3f}")
    typer.echo(f"📊 Test AP: {metrics['ap']:.3f}")


@app.command()
def eval_model(
    data_path: Path = typer.Option(Path("data/processed/fused.parquet"), help="Fused data file"),
    output_dir: Path = typer.Option(Path("reports"), help="Output directory"),
    model_type: str = typer.Option("logistic", help="Model type"),
    target_col: str = typer.Option("target", help="Target column name")
):
    """Evaluate model and save metrics."""
    setup_environment()
    
    typer.echo("📊 Evaluating model...")
    
    if not data_path.exists():
        typer.echo(f"❌ Data file not found: {data_path}")
        raise typer.Exit(1)
    
    # Load data
    data = pd.read_parquet(data_path)
    
    # Evaluate
    metrics = eval.evaluate_model_and_save(
        data,
        target_col=target_col,
        model_type=model_type,
        output_dir=output_dir,
        random_state=42
    )
    
    typer.echo(f"✅ Evaluation completed")
    typer.echo(f"📁 Results saved to {output_dir}")


@app.command()
def diagnostics_cmd(
    data_path: Path = typer.Option(Path("data/processed/fused.parquet"), help="Fused data file"),
    output_path: Path = typer.Option(Path("reports/leakage.json"), help="Output diagnostics file"),
    target_col: str = typer.Option("target", help="Target column name"),
    plate_col: str = typer.Option("plate_id", help="Plate ID column"),
    well_row_col: str = typer.Option("well_row", help="Well row column"),
    well_col_col: str = typer.Option("well_col", help="Well column")
):
    """Run leakage and batch diagnostics."""
    setup_environment()
    
    typer.echo("🔍 Running leakage diagnostics...")
    
    if not data_path.exists():
        typer.echo(f"❌ Data file not found: {data_path}")
        raise typer.Exit(1)
    
    # Load data
    data = pd.read_parquet(data_path)
    
    # Run diagnostics
    results = diagnostics.run_full_diagnostics(
        data,
        target=target_col,
        plate_id_col=plate_col,
        well_row_col=well_row_col,
        well_col_col=well_col_col,
        output_path=output_path,
        random_state=42
    )
    
    risk_level = results.get("leakage_risk", {}).get("overall_risk", "unknown")
    typer.echo(f"✅ Diagnostics completed")
    typer.echo(f"⚠️  Risk Level: {risk_level.upper()}")
    typer.echo(f"📁 Results saved to {output_path}")


@app.command()
def ic50(
    output_dir: Path = typer.Option(Path("reports"), help="Output directory"),
    n_compounds: int = typer.Option(3, help="Number of synthetic compounds"),
    n_points: int = typer.Option(8, help="Points per dose-response curve")
):
    """Run IC50 dose-response analysis."""
    setup_environment()
    
    typer.echo("💊 Running IC50 analysis...")
    
    # Generate synthetic dose-response data
    synthetic_data = dose_response.create_synthetic_dose_response_data(
        n_compounds=n_compounds,
        n_points_per_compound=n_points,
        random_state=42
    )
    
    # Process dose-response data
    results = dose_response.process_dose_response_data(
        synthetic_data,
        output_dir=output_dir,
        plot_curves=True
    )
    
    successful_fits = sum(1 for r in results.values() if r.get("fit_success", False))
    
    typer.echo(f"✅ IC50 analysis completed")
    typer.echo(f"📊 Successful fits: {successful_fits}/{len(results)}")
    typer.echo(f"📁 Results saved to {output_dir}")


@app.command()
def report(
    reports_dir: Path = typer.Option(Path("reports"), help="Reports directory"),
    docs_dir: Path = typer.Option(Path("docs"), help="Docs directory for GitHub Pages"),
    model_card_path: Path = typer.Option(Path("model_card.md"), help="Model card file")
):
    """Build reports and documentation."""
    setup_environment()
    
    typer.echo("📝 Building reports...")
    
    # Build all reports
    success = reporting.build_reports(reports_dir, docs_dir, model_card_path)
    
    if success:
        typer.echo(f"✅ Reports built successfully")
        typer.echo(f"📁 Reports: {reports_dir}")
        typer.echo(f"📁 Docs: {docs_dir}")
        typer.echo(f"🌐 View at: {reports_dir}/index.html")
    else:
        typer.echo("❌ Report generation failed")
        raise typer.Exit(1)


@app.command()
def all_pipeline(
    data_dir: Path = typer.Option(Path("data"), help="Data directory"),
    reports_dir: Path = typer.Option(Path("reports"), help="Reports directory"),
    docs_dir: Path = typer.Option(Path("docs"), help="Docs directory"),
    n_samples: int = typer.Option(100, help="Number of synthetic samples"),
    force: bool = typer.Option(False, help="Force regeneration of existing files")
):
    """Run the complete pipeline from start to finish."""
    setup_environment()
    
    typer.echo("🚀 Running complete CP-Tox-Mini pipeline...")
    typer.echo("")
    
    # Step 1: Download data
    typer.echo("Step 1/8: Download data")
    try:
        download(data_dir, validate=True)
    except typer.Exit:
        typer.echo("❌ Pipeline failed at download step")
        raise
    
    # Step 2: Compute features
    typer.echo("\nStep 2/8: Compute features")
    try:
        features(data_dir, n_samples=n_samples, force=force)
    except typer.Exit:
        typer.echo("❌ Pipeline failed at features step")
        raise
    
    # Step 3: Fuse modalities
    typer.echo("\nStep 3/8: Fuse modalities")
    try:
        fuse(force=force)
    except typer.Exit:
        typer.echo("❌ Pipeline failed at fusion step")
        raise
    
    # Step 4: Train model
    typer.echo("\nStep 4/8: Train model")
    try:
        train()
    except typer.Exit:
        typer.echo("❌ Pipeline failed at training step")
        raise
    
    # Step 5: Evaluate model
    typer.echo("\nStep 5/8: Evaluate model")
    try:
        eval_model(output_dir=reports_dir)
    except typer.Exit:
        typer.echo("❌ Pipeline failed at evaluation step")
        raise
    
    # Step 6: Run diagnostics
    typer.echo("\nStep 6/8: Run diagnostics")
    try:
        diagnostics_cmd()
    except typer.Exit:
        typer.echo("❌ Pipeline failed at diagnostics step")
        raise
    
    # Step 7: IC50 analysis
    typer.echo("\nStep 7/8: IC50 analysis")
    try:
        ic50(output_dir=reports_dir)
    except typer.Exit:
        typer.echo("❌ Pipeline failed at IC50 step")
        raise
    
    # Step 8: Generate reports
    typer.echo("\nStep 8/8: Generate reports")
    try:
        report(reports_dir, docs_dir)
    except typer.Exit:
        typer.echo("❌ Pipeline failed at reporting step")
        raise
    
    typer.echo("")
    typer.echo("🎉 Pipeline completed successfully!")
    typer.echo(f"📊 View results at: {reports_dir}/index.html")
    typer.echo(f"🌐 GitHub Pages ready at: {docs_dir}/index.html")


@app.command()
def validate(
    manifest_path: Path = typer.Option(Path("manifests/data_manifest.json"), help="Manifest file")
):
    """Validate data against manifest."""
    setup_environment()
    
    typer.echo("🔍 Validating data...")
    
    if not manifest_path.exists():
        typer.echo(f"❌ Manifest not found: {manifest_path}")
        typer.echo("Run 'cp-tox-mini download' first")
        raise typer.Exit(1)
    
    success = hashing.validate_manifest(manifest_path)
    
    if success:
        typer.echo("✅ Validation passed")
    else:
        typer.echo("❌ Validation failed")
        raise typer.Exit(1)


@app.command()
def status():
    """Show pipeline status and available outputs."""
    setup_environment()
    
    typer.echo("📊 CP-Tox-Mini Pipeline Status")
    typer.echo("=" * 40)
    
    # Check key files
    files_to_check = [
        ("Manifest", Path("manifests/data_manifest.json")),
        ("Features", Path("data/processed/features.parquet")),
        ("Fused Data", Path("data/processed/fused.parquet")),
        ("Model Metrics", Path("reports/model_metrics.json")),
        ("Leakage Report", Path("reports/leakage.json")),
        ("IC50 Results", Path("reports/ic50_summary.json")),
        ("Main Report", Path("reports/index.html")),
        ("Documentation", Path("docs/index.html"))
    ]
    
    for name, path in files_to_check:
        status_icon = "✅" if path.exists() else "❌"
        typer.echo(f"{status_icon} {name}: {path}")
    
    typer.echo("")
    typer.echo("🔧 To run the complete pipeline:")
    typer.echo("  cp-tox-mini all")
    typer.echo("")
    typer.echo("📖 For help with individual commands:")
    typer.echo("  cp-tox-mini --help")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()