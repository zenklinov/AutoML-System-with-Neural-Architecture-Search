# Architecture Diagram Specification

**Tool**: Draw.io / Excalidraw
**Goal**: Visualize the Distributed NAS System highlighting the separation of Control Plane vs. Compute Plane.

## 📐 Layout & Swimlanes

Create a vertical flow chart with 4 horizontal swimlanes (Containers).

### Lane 1: User / Interface (Top)
*   **Color**: Light Grey Implementation
*   **Role**: Entry point for the system.

### Lane 2: Control Plane (The "Brain")
*   **Color**: Light Blue Background
*   **Role**: Orchestration, Scheduling, and Metrics.
*   **Infrastructure**: CPU-Only Instance (Stable, On-Demand).

### Lane 3: Compute Plane (The "Muscle")
*   **Color**: Light Green Background
*   **Role**: Model Training and Evaluation.
*   **Infrastructure**: Scale-out GPU Cluster (Spot Instances, Preemptible).

### Lane 4: Storage Layer (Bottom)
*   **Color**: Light Yellow Background
*   **Role**: Persistence and Registry.

---

## 📦 Nodes (Components)

### In "User / Interface" Lane
1.  **Node: User CLI / API**
    *   *Type*: Actor / Terminal Icon
    *   *Label*: "Engineer / CI Pipeline"

### In "Control Plane" Lane
2.  **Node: REST API Gateway**
    *   *Type*: Rectangle
    *   *Label*: "Search Job Submission"
3.  **Node: Ray Head Node** (Group/Container)
    *   *Components Inside*:
        *   **Search Engine**: Label "Bayesian Optimization (TPE)"
        *   **Scheduler**: Label "ASHA / HyperBand"
        *   **State Manager**: Label "Trial Tracker"

### In "Compute Plane" Lane
4.  **Node: GPU Worker Group** (Stack of cards to imply many)
    *   *Label*: "Spot Instance Workers (GPU)"
    *   *Badge*: "Preemptible"
    *   *Inside Each Worker*:
        *   "Build Graph (Search Space)"
        *   "Train 1 Epoch"
        *   "Metric Report"

### In "Storage Layer" Lane
5.  **Node: Object Storage**
    *   *Type*: Cylinder / Database
    *   *Label*: "Checkpoints (S3 / MinIO)"
6.  **Node: Model Registry**
    *   *Type*: Database
    *   *Label*: "Best Models Only"

---

## 🔗 Connections (Arrows)

1.  **User -> API Gateway**
    *   *Label*: "Submit Config (JSON)"
    *   *Line*: Solid, Black

2.  **API Gateway -> Ray Head**
    *   *Label*: "Init Search"
    *   *Line*: Solid, Black

3.  **Ray Head (Scheduler) -> GPU Workers**
    *   *Label*: "Dispatch Config / Resume"
    *   *Line*: Solid, Blue (Control Signal)

4.  **GPU Workers -> Ray Head (Search Engine)**
    *   *Label*: "Report Metrics (Acc/Loss)"
    *   *Line*: Dashed, Green (Async Report)

5.  **GPU Workers -> Object Storage**
    *   *Label*: "Persist State (Every Epoch)"
    *   *Line*: Solid, Orange
    *   *Note*: Critical for Spot Recovery

6.  **Ray Head -> GPU Workers (Reverse)**
    *   *Label*: "STOP Signal (Pruning)"
    *   *Line*: Solid, Red (Termination)

7.  **Ray Head -> Model Registry**
    *   *Label*: "Finalize Best Model"
    *   *Line*: Solid, Purple

---

## 📝 Loop & Failure Logic (Annotations)

*   **Spot Death Loop**: Add a self-loop arrow on "Control Plane" labeled *"Worker Lost -> Reschedule on New Node"*.
*   **Pruning**: Add annotation on the Red Line: *"Bottom 70% pruned early"*.
