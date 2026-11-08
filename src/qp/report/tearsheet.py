"""Tearsheet: a PNG (equity, drawdown, rolling Sharpe, exposure, monthly heatmap)
plus a plain-text metrics table. No seaborn; matplotlib only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..backtest import metrics as M  # noqa: E402
from ..backtest.engine import BacktestResult  # noqa: E402


def _monthly_table(returns: pd.Series) -> pd.DataFrame:
    m = (1 + returns).resample("ME").prod() - 1
    df = m.to_frame("r")
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.pivot(index="year", columns="month", values="r")


def write_tearsheet(result: BacktestResult, outdir: str | Path, title: str = "strategy") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    r = result.returns.dropna()
    eq = M.equity_curve(r)
    dd = M.drawdown(r)
    roll_sharpe = (
        r.rolling(126).mean() / r.rolling(126).std() * np.sqrt(252)
    )
    gross_exp = result.weights.abs().sum(axis=1)

    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle(f"{title}  --  Sharpe {result.metrics['sharpe']:.2f}, "
                 f"CAGR {result.metrics['cagr']*100:.1f}%, "
                 f"maxDD {result.metrics['max_drawdown']*100:.1f}%", fontsize=13)

    ax[0, 0].plot(eq.index, eq.values, lw=1.2)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("Equity curve (log)")

    ax[0, 1].fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax[0, 1].set_title("Drawdown")

    ax[1, 0].plot(roll_sharpe.index, roll_sharpe.values, lw=1.0)
    ax[1, 0].axhline(0, color="k", lw=0.6)
    ax[1, 0].set_title("Rolling 6m Sharpe")

    ax[1, 1].plot(gross_exp.index, gross_exp.values, lw=0.8)
    ax[1, 1].plot(result.weights.sum(axis=1).index, result.weights.sum(axis=1).values,
                  lw=0.8, alpha=0.7)
    ax[1, 1].legend(["gross", "net"], fontsize=8)
    ax[1, 1].set_title("Exposure")

    comp = result.pnl_components[["gross", "transaction_cost", "borrow_cost"]].cumsum()
    ax[2, 0].plot(comp.index, comp.values, lw=0.9)
    ax[2, 0].legend(["cum gross", "cum tcost", "cum borrow"], fontsize=8)
    ax[2, 0].set_title("Cumulative P&L decomposition")

    mt = _monthly_table(r)
    if not mt.empty:
        im = ax[2, 1].imshow(mt.values, aspect="auto", cmap="RdYlGn",
                             vmin=-abs(np.nanmax(np.abs(mt.values))),
                             vmax=abs(np.nanmax(np.abs(mt.values))))
        ax[2, 1].set_xticks(range(mt.shape[1]))
        ax[2, 1].set_xticklabels(mt.columns)
        ax[2, 1].set_yticks(range(mt.shape[0]))
        ax[2, 1].set_yticklabels(mt.index)
        ax[2, 1].set_title("Monthly returns")
        fig.colorbar(im, ax=ax[2, 1], fraction=0.046)

    for a in ax.flat:
        a.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = outdir / "tearsheet.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)

    (outdir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    _write_text_report(result, outdir / "report.txt", title)
    r.to_frame("net_return").to_csv(outdir / "returns.csv")
    result.weights.to_parquet(outdir / "weights.parquet")
    return png


def _write_text_report(result: BacktestResult, path: Path, title: str) -> None:
    m = result.metrics
    lines = [
        f"# {title}",
        "",
        f"{'metric':<22}{'value':>14}",
        "-" * 36,
        f"{'days':<22}{m['n_days']:>14d}",
        f"{'CAGR':<22}{m['cagr']*100:>13.2f}%",
        f"{'ann vol':<22}{m['ann_vol']*100:>13.2f}%",
        f"{'Sharpe':<22}{m['sharpe']:>14.2f}",
        f"{'Sortino':<22}{m['sortino']:>14.2f}",
        f"{'max drawdown':<22}{m['max_drawdown']*100:>13.2f}%",
        f"{'Calmar':<22}{m['calmar']:>14.2f}",
        f"{'hit rate':<22}{m['hit_rate']*100:>13.2f}%",
        f"{'skew':<22}{m['skew']:>14.2f}",
        f"{'kurtosis':<22}{m['kurtosis']:>14.2f}",
        f"{'daily VaR 95':<22}{m['var_95']*100:>13.2f}%",
        f"{'daily CVaR 95':<22}{m['cvar_95']*100:>13.2f}%",
    ]
    if "avg_daily_turnover" in m:
        lines += [
            f"{'avg gross exposure':<22}{m['avg_gross_exposure']:>14.2f}",
            f"{'avg net exposure':<22}{m['avg_net_exposure']:>14.2f}",
            f"{'avg daily turnover':<22}{m['avg_daily_turnover']:>14.2f}",
            f"{'time in market':<22}{m['time_in_market']*100:>13.2f}%",
        ]
    path.write_text("\n".join(lines) + "\n")
