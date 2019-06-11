# Application of TPOT's new Feature Set Selector (FSS) and Template

This repository contains detailed simulation and analysis code needed to reproduce the results in this study: [Scaling tree-based automated machine learning to biomedical big data with a feature set selector](https://doi.org/10.1093/bioinformatics/btz470).

Simulation analyses and visualizations are in [simulation](https://github.com/lelaboratoire/tpot-fss/tree/master/simulation), and real-world application to RNA-Seq data are in [RNASeq](https://github.com/lelaboratoire/tpot-fss/tree/master/RNASeq).
Each folder contains TPOT exported pipelines (with FSS in `pipelines_ds` and without FSS in `pipelines_reg`) as well as the cross validated accuracy of each pipeline (in `accuracies_*`).

Questions/PRs are welcomed.
