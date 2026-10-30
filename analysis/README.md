# Contact-map and P(s) analysis

## Experimental data

scsHi-C data are from Mitter et al. 2020 (GEO GSE152373), as processed
`.cool` files at 1 kb resolution. Place them in `../experimental_data/`
(set `EXP_ROOT` in `P_s_example.ipynb`).

| Condition | Figure | GEO | GSM |
|---|---|---| --|
| WT G₂-16h | 1, 3 | GSE 152373 | GSM4613674 |
| G₂ entry (WT G₂-0 h / S8h) | 1 | GSE 152373 | GSM4613673 |
| ΔSORORIN G₂-16 h | 1, 3 | GSE 152373 | GSM4613677 |
| ΔWAPL G₂-16 h | 1, 4 | GSE 227694 | GSM7105758-GSM7105763 |
| ΔSORORIN+ΔWAPL G₂-16 h | 1, 4 | GSE 227694 |GSM7105764–GSM7105767 |
| S2h | S11 | GSE 152373 | GSM4613671 |
| S4h | S11 | GSE 152373 | GSM4613672 |


Genome assembly: [hg19/hg38]. Replicates are pooled as in the GEO
processed files; all autosomes are aggregated.

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
