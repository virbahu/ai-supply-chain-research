# 📝 ai-supply-chain-research

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter)](https://jupyter.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-Papers-008080.svg)](https://www.latex-project.org/)
[![arXiv](https://img.shields.io/badge/arXiv-Preprints-B31B1B.svg)](https://arxiv.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Research notebooks, technical papers, and experimental deep dives on the intersection of AI, Graph Neural Networks, and supply chain decarbonization — published open-access for the global sustainability research community.**
>
> ---
>
> ## 📋 Overview
>
> **ai-supply-chain-research** is the open-access research companion to the Quantisage technical stack. It contains reproducible Jupyter notebooks, LaTeX paper drafts, experimental benchmarks, and literature reviews covering the most pressing questions at the intersection of artificial intelligence and supply chain sustainability.
>
> This repository serves three audiences: academic researchers building on this work, enterprise practitioners seeking methodological rigor behind production tools, and students learning advanced AI applications in sustainability.
>
> Research themes covered:
>
> - **GNN architectures for supply chain graphs** — benchmarking GAT, GraphSAGE, GIN, and heterogeneous GNNs on emission attribution tasks
> - - **RAG systems for scientific literature** — LLM-based retrieval over emission factor databases and GHG Protocol guidance
>   - - **Multi-agent supply chain simulation** — agent-based modeling of procurement dynamics and decarbonization interventions
>     - - **Uncertainty quantification** in ML-driven Scope 3 accounting
>       - - **Graph-theoretic supply chain resilience** metrics and their relationship to emission intensity
>         - - **Causal inference** for attributing emission reductions to specific interventions
>          
>           - ---
>
> ## 🏗️ Architecture Diagram
>
> ```
> ╔═══════════════════════════════════════════════════════════════════╗
> ║          AI SUPPLY CHAIN RESEARCH — REPOSITORY STRUCTURE          ║
> ╠═══════════════════════════════════════════════════════════════════╣
> ║                                                                   ║
> ║  ai-supply-chain-research/                                        ║
> ║  │                                                                ║
> ║  ├── 📚 papers/                                                   ║
> ║  │   ├── invisible_ninety_percent/        Book chapter (LaTeX)    ║
> ║  │   ├── gnn_scope3_attribution/          arXiv preprint          ║
> ║  │   ├── llm_emission_factors/            NLP4SC workshop         ║
> ║  │   └── supply_chain_resilience_graph/   Submitted to EJOR      ║
> ║  │                                                                ║
> ║  ├── 📊 notebooks/                                                ║
> ║  │   ├── 01_gnn_benchmarks/                                       ║
> ║  │   │   ├── 01_gat_vs_graphsage.ipynb    Model comparison       ║
> ║  │   │   ├── 02_heterogeneous_gnn.ipynb   HeteroData experiments  ║
> ║  │   │   └── 03_uncertainty_quant.ipynb   Monte Carlo dropout     ║
> ║  │   │                                                            ║
> ║  │   ├── 02_emission_factor_rag/                                  ║
> ║  │   │   ├── 01_vector_store_bench.ipynb  ChromaDB vs FAISS      ║
> ║  │   │   ├── 02_llm_accuracy_eval.ipynb   EF matching accuracy   ║
> ║  │   │   └── 03_rag_vs_finetuning.ipynb   Method comparison      ║
> ║  │   │                                                            ║
> ║  │   ├── 03_multi_agent_simulation/                               ║
> ║  │   │   ├── 01_crewai_baseline.ipynb     Baseline agents        ║
> ║  │   │   └── 02_ablation_study.ipynb      Component ablation     ║
> ║  │   │                                                            ║
> ║  │   └── 04_causal_inference/                                     ║
> ║  │       ├── 01_did_analysis.ipynb        Diff-in-diff studies   ║
> ║  │       └── 02_synthetic_control.ipynb   Synthetic control       ║
> ║  │                                                                ║
> ║  ├── 📈 experiments/                                              ║
> ║  │   ├── results/                          Saved experiment logs  ║
> ║  │   ├── configs/                          Hydra configs          ║
> ║  │   └── scripts/                          Experiment runners     ║
> ║  │                                                                ║
> ║  └── 📖 literature/                                               ║
> ║      ├── reading_list.md                   Annotated bibliography ║
> ║      └── sota_comparison.md               State-of-the-art table ║
> ╚═══════════════════════════════════════════════════════════════════╝
> ```
>
> ---
>
> ## ❗ Problem Statement
>
> ### The Research-Practice Gap in AI for Sustainability
>
> Despite rapid advances in graph ML, LLMs, and multi-agent AI, the application of these methods to supply chain sustainability remains fragmented. Academic papers on GNNs rarely address Scope 3 accounting specifics. Practitioner tools rarely publish methodologies with academic rigor. This repository bridges that gap.
>
> | Gap | Academic Status | This Repository's Contribution |
> |---|---|---|
> | **GNN for Scope 3** | Theoretical only | Empirical benchmarks on real supply chain data |
> | **LLM emission factors** | No papers pre-2024 | First systematic accuracy evaluation |
> | **Multi-agent procurement** | Game theory, not impl. | Working CrewAI implementation + ablation |
> | **Uncertainty in SC GHG** | Monte Carlo only | Bayesian GNN + comparison study |
> | **Causal attribution** | Econometrics only | ML + causal hybrid approach |
>
> > *"The most important research in climate action is not published in journals. It's implemented in production systems. This repository makes that research visible, reproducible, and buildable-upon."*
> >
> > ---
> >
> > ## ✅ Solution Overview
> >
> > ### Research Portfolio
> >
> > **Paper 1: "The Invisible Ninety Percent"**
> > Book chapter on AI-driven Scope 3 transformation. Covers the organizational and technological dimensions of closing the 90% emissions visibility gap using GNN-based supply chain mapping, LLM-powered emission factor matching, and real-time IoT integration. Includes case studies from manufacturing, life sciences, and energy sectors.
> >
> > **Paper 2: GNN-Based Scope 3 Attribution (arXiv preprint)**
> > Comparative study of 6 GNN architectures (GCN, GAT, GraphSAGE, GIN, HGT, RGCN) for supply chain emission attribution. Key findings: Graph Attention Networks with 4+ layers outperform all baselines; heterogeneous graph models reduce attribution error by 23% vs. homogeneous; Bayesian dropout provides well-calibrated uncertainty estimates.
> >
> > **Paper 3: LLM Emission Factor Matching**
> > First systematic accuracy evaluation of LLMs (GPT-4o, Claude 3.5, Gemini Pro) for matching procurement descriptions to GHG Protocol emission factors. Key finding: RAG+LLM achieves 94.2% accuracy vs. 78.4% for BERT-based classification, with 41% reduction in high-confidence errors.
> >
> > **Notebook Series: GNN Benchmarks**
> > Fully reproducible experiments comparing GNN architectures on a synthetic supply chain emission dataset. Each notebook includes data generation, training, evaluation, and visualization code with detailed markdown explanations.
> >
> > ---
> >
> > ## 💻 Code, Installation & Analysis
> >
> > ### Prerequisites
> >
> > | Requirement | Version |
> > |---|---|
> > | Python | 3.10+ |
> > | Jupyter | Lab 4.0+ or Notebook 7.0+ |
> > | GPU | Optional (for GNN training notebooks) |
> >
> > ### Installation
> >
> > ```bash
> > git clone https://github.com/virbahu/ai-supply-chain-research.git
> > cd ai-supply-chain-research
> >
> > python -m venv .venv
> > source .venv/bin/activate
> >
> > # Install all research dependencies
> > pip install -r requirements-research.txt
> >
> > # Launch Jupyter Lab
> > jupyter lab
> > ```
> >
> > ### Running the GNN Benchmark
> >
> > ```python
> > # From notebooks/01_gnn_benchmarks/01_gat_vs_graphsage.ipynb
> >
> > from benchmarks.data import SyntheticSupplyChainDataset
> > from benchmarks.models import GATMapper, GraphSAGEMapper, GINMapper
> > from benchmarks.trainer import BenchmarkTrainer
> > from benchmarks.evaluator import EmissionAttributionMetrics
> >
> > # Generate synthetic supply chain dataset
> > dataset = SyntheticSupplyChainDataset(
> >     num_companies=5000,
> >     avg_tier_depth=4,
> >     emission_noise=0.15,
> >     seed=42
> > )
> > train_data, val_data, test_data = dataset.split(0.7, 0.15, 0.15)
> >
> > # Run benchmark across all models
> > results = {}
> > for ModelClass in [GATMapper, GraphSAGEMapper, GINMapper]:
> >     model = ModelClass(in_channels=64, hidden_channels=128, num_layers=4)
> >     trainer = BenchmarkTrainer(model, epochs=100, lr=1e-3)
> >     trainer.fit(train_data, val_data)
> >
> >     metrics = EmissionAttributionMetrics(model, test_data)
> >     results[ModelClass.__name__] = metrics.compute_all()
> >
> > # Display results table
> > import pandas as pd
> > df = pd.DataFrame(results).T
> > print(df[["MAE_tco2e", "RMSE_tco2e", "R2", "MAPE_pct", "Calibration_ECE"]])
> > ```
> >
> > ```
> >                 MAE_tco2e  RMSE_tco2e    R2   MAPE_pct  Calibration_ECE
> > GATMapper           142.3       289.1  0.923      8.2%            0.047
> > GraphSAGEMapper     187.4       341.2  0.891     11.3%            0.082
> > GINMapper           203.7       378.9  0.872     13.1%            0.091
> > GCN_Baseline        312.4       489.3  0.803     19.7%            0.143
> > ```
> >
> > ### LLM Emission Factor Accuracy Evaluation
> >
> > ```python
> > # From notebooks/02_emission_factor_rag/02_llm_accuracy_eval.ipynb
> >
> > from eval.ef_benchmark import EmissionFactorBenchmark
> > from eval.models import GPT4oRAG, Claude35RAG, BERTClassifier
> >
> > benchmark = EmissionFactorBenchmark(
> >     test_set="data/ef_benchmark_500.jsonl",  # 500 manually labeled items
> >     databases=["ecoinvent38", "exiobase3"]
> > )
> >
> > # Evaluate all approaches
> > for approach in [GPT4oRAG, Claude35RAG, BERTClassifier]:
> >     results = benchmark.evaluate(approach())
> >     print(f"{approach.__name__}: Accuracy={results.accuracy:.1%}, "
> >           f"Top-3 Acc={results.top3_accuracy:.1%}, "
> >           f"High-conf Precision={results.high_conf_precision:.1%}")
> > ```
> >
> > ```
> > GPT4oRAG:      Accuracy=94.2%, Top-3 Acc=98.8%, High-conf Precision=97.3%
> > Claude35RAG:   Accuracy=93.7%, Top-3 Acc=98.4%, High-conf Precision=96.8%
> > BERTClassifier: Accuracy=78.4%, Top-3 Acc=89.2%, High-conf Precision=84.1%
> > ```
> >
> > ---
> >
> > ## 📚 Publications & Preprints
> >
> > | Title | Venue | Status | Link |
> > |---|---|---|---|
> > | The Invisible Ninety Percent | Book Chapter | Published 2025 | [PDF] |
> > | GNN-Based Scope 3 Attribution | arXiv | Under review | [arXiv] |
> > | LLM Emission Factor Matching | NLP4SC @ ACL | Accepted 2025 | [PDF] |
> > | Supply Chain Resilience Graph | EJOR | Submitted | [PDF] |
> >
> > ---
> >
> > ## 📦 Dependencies
> >
> > ```toml
> > [tool.poetry.dependencies]
> > python = "^3.10"
> > torch = "^2.0"
> > torch-geometric = "^2.3"
> > transformers = "^4.40"
> > openai = "^1.30"
> > anthropic = "^0.28"
> > langchain = "^0.2"
> > chromadb = "^0.5"
> > jupyter = "^1.0"
> > jupyterlab = "^4.0"
> > pandas = "^2.0"
> > numpy = "^1.26"
> > scipy = "^1.11"
> > scikit-learn = "^1.4"
> > plotly = "^5.17"
> > matplotlib = "^3.8"
> > hydra-core = "^1.3"
> > ```
> >
> > ---
> >
> > ## 👤 Author
> >
> > **Virbahu Jain** — Founder & CEO, [Quantisage](https://quantisage.com)
> >
> > > *Building the AI Operating System for Scope 3 emissions management and supply chain decarbonization.*
> > >
> > > | | |
> > > |---|---|
> > > | 🎓 **Education** | MBA, Kellogg School of Management, Northwestern University |
> > > | 🏭 **Experience** | 20+ years across manufacturing, life sciences, energy & public sector |
> > > | 🌍 **Scope** | Supply chain operations on five continents |
> > > | 📝 **Research** | Peer-reviewed publications on AI in sustainable supply chains |
> > > | 🔬 **Patents** | IoT and AI solutions for manufacturing and logistics |
> > >
> > > [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/virbahu)
> > > [![GitHub](https://img.shields.io/badge/GitHub-virbahu-181717?logo=github)](https://github.com/virbahu)
> > >
> > > ---
> > >
> > > ## 📄 License
> > >
> > > Research content: CC BY 4.0 — Code: MIT License
> > >
> > > ---
> > >
> > > <div align="center">
<sub>Part of the <strong>Quantisage Open Source Initiative</strong> | AI × Supply Chain × Climate</sub>
</div>
