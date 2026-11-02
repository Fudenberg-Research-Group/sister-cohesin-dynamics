# Contact-map and P(s) analysis

## Experimental data

scsHi-C data are from Mitter et al. 2020 (GEO GSE152373) and Batty et
al. 2023 (GEO GSE227694).

| Condition | Figure | GEO | GSM |
|---|---|---|---|
| WT G₂-16 h | 1, 3 | GSE152373 | GSM4613674 |
| G₂ entry (WT G₂-0 h / S8h) | 1 | GSE152373 | GSM4613673 |
| ΔSORORIN G₂-16 h | 1, 3 | GSE152373 | GSM4613677 |
| ΔWAPL G₂-16 h | 1, 4 | GSE227694 | GSM7105758–GSM7105763 |
| ΔSORORIN+ΔWAPL G₂-16 h | 1, 4 | GSE227694 | GSM7105764–GSM7105767 |
| S2h | S11 | GSE152373 | GSM4613671 |
| S4h | S11 | GSE152373 | GSM4613672 |

Replicate-level cis and trans pairs were converted
to 1 kb COOL files and raw counts were summed across six ΔWAPL replicates and
four ΔSORORIN+ΔWAPL replicates. The pooled files are
`dW_merged.cis.1kb.cool`, `dW_merged.trans.1kb.cool`,
`dWdS_merged.cis.1kb.cool`, and `dWdS_merged.trans.1kb.cool`.

Genome assembly: hg38.

## Running

Run from `analysis/`:

```bash
cd analysis
python contacts_vs_distance_from_trajectories.py 1
```

The argument is a one-based job index into `ALPHA_LIST × TAU_LIST` (with the
defaults there is a single job, index 1), so it can be passed directly as
`$SLURM_ARRAY_TASK_ID` with `--array=1-N`. Replica indices elsewhere in the
repository are zero-based. `P_s_example.ipynb` is also run from `analysis/`.

## Parameter inference

`inference_helpers.py` provides the shared combined, cis-only, and trans-only
log-space R-squared calculations and parameter-grid scoring used for
manuscript inference. The helper operates on P(s) curves that have already
been extracted from experimental and simulated contact maps; it does not
perform contact extraction itself.

The scoring definition, expected dictionary structure, and the verified
G2-0 h sister-configuration result used in Figure S2 are documented in
[`PARAMETER_INFERENCE.md`](PARAMETER_INFERENCE.md).

## P(s) signature metrics

`signature_metrics.py` calculates three metrics from extracted P(s) curves. The
gap is the dimensionless cis/trans P(s) ratio at 40 kb; the slope is the signed
local log-log slope of trans P(s) at 40 kb; and the shoulder is the genomic
distance at which the cis P(s) log-log derivative is maximal within the chosen
search window. Distances returned by the helper are in bp.

## Memory requirements

`contacts_vs_distance_from_trajectories.py` currently accumulates dense `int32` contact matrices in
memory. With the default system size of 320,000 monomers per chromatid and
8 monomers per bin, each 40,000 × 40,000 matrix requires approximately
6.4 GB (5.96 GiB).

Each cis-contact worker allocates two such matrices. Processing five
replicas can therefore require substantially more than 64 GB once worker
matrices, parent-process aggregation, multiprocessing copies, coordinates,
and contact lists are included. Run this analysis on a high-memory compute
node.

Increasing `NUMBER_OF_MAPS` reduces memory use by using coarser bins, but
also changes the contact-map resolution. Sparse or blockwise accumulation
would be preferable for future lower-memory implementations.
