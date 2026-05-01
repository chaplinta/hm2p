"""Stage 5b — sync report aggregator → sync_report.parquet.

Reads root attrs from every sync.h5 in derivatives/sync/ and writes a
single parquet file with one row per session. See
docs/sync-pipeline-design.md §1.4 for the column schema.

Snakemake wires this rule downstream of every per-session sync rule, so
the DAG guarantees that no aggregation runs while Stage 5 is mid-flight.
"""


rule sync_report:
    """Aggregate per-session sync.h5 attrs → sync_report.parquet."""
    input:
        sync_files=expand(
            f"{DATA_ROOT}/derivatives/sync/{{sub}}/{{ses}}/sync.h5",
            zip,
            sub=[sub(s) for s in SESSIONS],
            ses=[ses(s) for s in SESSIONS],
        ),
    output:
        parquet=f"{DATA_ROOT}/derivatives/sync_report/sync_report.parquet",
    container:
        cpu_container()
    resources:
        mem_mb=2000,
        runtime=5,
    shell:
        """
        python -c "
from pathlib import Path
from hm2p.sync.report import build_report
build_report(
    Path('{DATA_ROOT}/derivatives/sync'),
    Path('{output.parquet}'),
)
"
        """
