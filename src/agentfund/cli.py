"""Command line interface: run, probe-leakage, report."""

from __future__ import annotations

from pathlib import Path

import typer

from .config import root

app = typer.Typer(add_completion=False, help="Multi-agent LLM hedge fund with lookahead defences.")
Backend_ = typer.Option(
    "ollama", help="ollama (default, local) or anthropic (needs ANTHROPIC_API_KEY)"
)


@app.command()
def run(
    start: str = typer.Option("2024-01-01", help="first decision month, YYYY-MM-DD"),
    end: str = typer.Option("2026-08-31", help="last day of data used"),
    backend: str = Backend_,
    model: str = typer.Option(None, help="model name; defaults per backend"),
    anonymise: bool = typer.Option(True, "--anonymise/--no-anonymise"),
    cost_bps: float = typer.Option(10.0, help="cost per side in basis points"),
    out: Path = typer.Option(None, help="output directory (default results/run)"),
):
    """Walk-forward backtest of the agents, ablations and baselines."""
    from .llm import make_backend
    from .runner import run_backtest

    res = run_backtest(
        start,
        end,
        make_backend(backend, model),
        out or root() / "results" / "run",
        anonymise,
        cost_bps,
        log=typer.echo,
    )
    typer.echo(f"Sharpe: { ({k: round(v['sharpe'], 2) for k, v in res['metrics'].items()}) }")


@app.command("probe-leakage")
def probe_leakage(
    backend: str = Backend_,
    model: str = typer.Option(None),
    out: Path = typer.Option(None, help="default results/probe.json"),
):
    """Ask the model to recall who/when/what-next, raw vs anonymised, before vs after cutoff."""
    from .llm import make_backend
    from .probe import run_probe

    res = run_probe(
        make_backend(backend, model), out or root() / "results" / "probe.json", log=typer.echo
    )
    for r in res["table"]:
        typer.echo(
            f"{r['mode']:10s} {r['window']:4s} ticker {r['id_correct']['rate']:.1%} "
            f"direction {r['direction_correct']['rate']:.1%} (n={r['n']})"
        )


@app.command()
def report(
    results: Path = typer.Option(None, help="results directory (default ./results)"),
    docs: Path = typer.Option(None, help="figure directory (default ./docs)"),
):
    """Render figures (docs/) and markdown tables from saved results."""
    from . import report as rp

    results, docs = results or root() / "results", docs or root() / "docs"
    docs.mkdir(exist_ok=True)
    run_dir, probe = results / "run", results / "probe.json"
    if (run_dir / "summary.json").exists():
        if (run_dir / "equity.csv").exists():  # not committed: re-create with `agentfund run`
            rp.equity_figure(run_dir, docs / "equity_curves.png")
        (docs / "results_table.md").write_text(rp.results_table(run_dir) + "\n")
        (docs / "example_trace.md").write_text(rp.example_trace(run_dir) + "\n")
        typer.echo(rp.results_table(run_dir))
    if probe.exists():
        rp.probe_figure(probe, docs / "leakage_probe.png")
        (docs / "probe_table.md").write_text(rp.probe_table(probe) + "\n")
        typer.echo(rp.probe_table(probe))
