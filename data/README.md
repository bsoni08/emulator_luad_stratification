# Input data

Not included in the repository. The training script expects:

| File | Content |
|---|---|
| `simulated_gene_expression_file_heter.csv` | Simulated bulk expression, genes in rows and samples in columns |
| `decon_signature_proportions_file_heter.csv` | TOAST proportions of Groups 0–3, samples in rows |
| `group_signature_genes.csv` | Marker genes, one column per group (`Group_0` … `Group_3`) |
| `reference_bulk_expression.csv` | A clinical bulk expression table, genes in rows; only its gene names are used, to keep the genes shared with clinical data |
