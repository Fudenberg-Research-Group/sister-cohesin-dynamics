# Contact-map and P(s) analysis

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
