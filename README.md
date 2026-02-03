# Enterprise AutoML & Neural Architecture Search (NAS) System

> **Production-Grade, Distributed, and Cost-Aware Architecture Search Platform**

> **Architecture Artifacts**:  
> [Open Diagram (draw.io)](architecture/system_design.drawio) ·  
> [Diagram Specification](architecture/diagram_spec.md)

---

## Purpose

Most AutoML and NAS repositories focus on algorithmic novelty.  
This repository focuses on **system design and operational reality**.

Designing a NAS algorithm is straightforward. Designing a NAS **system** that operates reliably under cost constraints, infrastructure volatility, and distributed execution is not.

This project demonstrates how to:

- **Orchestrate Distributed Search**: Scale from a single GPU to large clusters using Ray Tune without modifying search logic.
- **Control Compute Spend**: Reduce GPU cost by ~70% through Spot Instances, early termination, and parallel scheduling.
- **Operate Under Failure**: Recover from preemption, OOMs, and partial node loss without corrupting search state.
- **Separate Concerns Cleanly**: Isolate orchestration (CPU), training (GPU), and state (storage).

This repository is a **reference architecture**, not an academic experiment.

---

## Scope & Non-Goals

To avoid misinterpretation, the following are explicitly out of scope:

- **Managed SaaS Platform**: This is not a multi-tenant, hosted AutoML service.
- **Feature Engineering Automation**: The focus is neural architecture search, not data preprocessing.
- **Single-Node Optimization**: While local execution is supported, the architecture is justified only at distributed scale.

---

## System Architecture

The system follows a **split-plane design**, separating low-cost control logic from high-cost compute execution.

### Control Plane (Stable, Low Cost)

- **Ray Head Node**  
  Runs on a CPU-only instance (e.g., `t3.medium`) and maintains global search state.

- **Search Engine (Bayesian Optimization)**  
  Implemented using Optuna (TPE) to sample promising architectures efficiently.

- **Scheduler (ASHA / HyperBand)**  
  Continuously evaluates trial performance and aggressively terminates underperforming candidates to preserve budget.

### Compute Plane (Ephemeral, Cost-Optimized)

- **GPU Workers**  
  Stateless training workers designed explicitly for Spot / Preemptible instances.

- **Failure Recovery**  
  When a worker is preempted, the orchestrator reschedules the trial and restores weights from the latest checkpoint.

### Execution Flow

1. A user submits a NAS job via CLI or API.
2. The orchestrator initializes candidate architectures.
3. The scheduler dispatches trials to available GPU workers.
4. Workers train for short intervals, report metrics, and receive continuation or termination signals.
5. Only the top-performing architectures are persisted to the model registry.

---

## Observability & Monitoring

Operational visibility is treated as a first-class concern:

- **Per-Trial Telemetry**  
  Ray Tune captures metrics (loss, accuracy), logs, and failure signals for each trial.

- **Centralized Experiment Tracking**  
  Integration with MLflow and TensorBoard enables comparison between completed and pruned architectures.

- **Cost Attribution**  
  Each trial is tagged with runtime duration and node type, enabling precise GPU-hour accounting.  
  Example output is provided in `results/sample_metrics.csv`.

---

## Neural Architecture Search Design

This system performs **true architecture search**, not parameter tuning.

### Search Space

- **Convolutional Blocks**
  - Kernel sizes: `{3×3, 5×5, 7×7}`
  - Expansion ratios: `{1, 2, 4}`

- **Attention Blocks**
  - Heads: `{2, 4, 8}`
  - MLP ratios: `{2, 4}`

- **Depth Control**
  - Dynamic depth between 8 and 32 layers using stochastic depth sampling.

### Search Strategy

- **Bayesian Optimization (TPE)**  
  Prioritizes promising regions of the search space and converges significantly faster than random search.

- **ASHA Early Stopping**  
  Implements a strict “fail fast” policy, terminating poorly performing architectures after minimal compute investment.

---

## Cost Analysis

A naïve NAS implementation on ImageNet-scale workloads can cost hundreds of dollars.  
This architecture is designed to reduce that cost by an order of magnitude.

### Baseline Configuration

- Instance: AWS `g4dn.xlarge` (On-Demand)
- Trials: 50
- Epochs per trial: 20
- Cost per hour: $0.526  
- **Estimated total**: ~$52.60

### Optimized Configuration (This System)

- Instance: AWS Spot `g4dn.xlarge`
- Average Spot price: ~$0.158/hour
- Early termination rate: ~70%
- Parallel execution eliminates idle time  
- **Estimated total**: **< $5.00**

**Business impact**: Comparable model quality achieved at <10% of the cost of a full grid search.

---

## Failure & Recovery Model

Failure is assumed and explicitly designed for.

| Failure Scenario | Mitigation |
|------------------|------------|
| GPU OOM | Trial is marked as pruned and excluded from future sampling. |
| Spot preemption | Trial is rescheduled automatically and resumed from checkpoint. |
| Orchestrator restart | Search state is persisted and resumes without manual intervention. |

---

## Security Considerations

- **Network Isolation**  
  The Ray cluster is intended to run inside a private VPC with no public ingress.

- **Secrets Management**  
  Cloud credentials are provided via IAM roles or environment variables, never embedded in code.

- **Serialization Trust Model**  
  PyTorch/Ray serialization assumes trusted internal execution; untrusted artifact ingestion is out of scope.

---

## Repository Structure

```bash
AutoML-System-with-Neural-Architecture-Search/
├── architecture/
│   ├── system_design.drawio
│   └── diagram_spec.md
├── src/
│   ├── orchestrator/
│   ├── nas/
│   ├── training/
│   └── evaluation/
├── config/
│   ├── search_space.yaml
│   └── runtime.yaml
└── scripts/
```

## Getting Started

### Prerequisites
*   Python 3.9+
*   CUDA 11.x (or CPU mode for testing)
*   Ray Cluster (Local or K8s)

### Execution
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Search (Local Mode)
./scripts/run_search.sh bayesian 20
```

---
