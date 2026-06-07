"""
ContinualExperimentManager
==========================
This is Task 2: the continual learning experiment setup.

It manages:
  - The performance matrix (rows = learning step, columns = task being evaluated)
  - Checking if the clusterer correctly found the new class (Task 3 lite)
  - Expanding the classifier when a new class arrives
  - Tracking metrics: Average Accuracy and Forgetting Measure

How the performance matrix works:
    After training on task T, we evaluate on ALL test splits seen so far (0..T).
    We write those accuracy scores into row T of the matrix.

    Example with 4 tasks:

              Task0  Task1  Task2  Task3
    Step 0  [ 0.95,  0.00,  0.00,  0.00 ]   ← trained on task0, tested on task0 only
    Step 1  [ 0.88,  0.91,  0.00,  0.00 ]   ← trained on task1, tested on task0 AND task1
    Step 2  [ 0.82,  0.87,  0.93,  0.00 ]   ← trained on task2, tested on all three
    Step 3  [ 0.79,  0.84,  0.90,  0.95 ]   ← trained on task3, tested on all four

    The diagonal = accuracy right after learning that task (plasticity).
    Off-diagonal drops = forgetting (stability).
"""

import numpy as np


class ContinualExperimentManager:
    def __init__(self, num_tasks, use_intervention_override=True):
        """
        Args:
            num_tasks (int):
                How many sequential datasets/bot-types you'll train on in total.
                This determines the size of the performance matrix.

            use_intervention_override (bool):
                Max's requested toggle flag (Task 3 lite).
                - True  → if the clusterer fails to detect the new class,
                          we manually force the classifier to expand anyway.
                - False → if the clusterer fails, we do nothing and let the
                          model try to cope (will likely hurt performance).
        """
        # The 2D performance matrix.
        # performance_matrix[step][task] = accuracy of the model at 'step' on 'task'.
        self.performance_matrix = np.full((num_tasks, num_tasks), np.nan)

        self.num_tasks = num_tasks
        self.current_step = 0          # which learning step we're on (0-indexed)
        self.use_intervention_override = use_intervention_override

        # History of per-task metrics, stored as plain dicts for easy inspection
        self.history = []

    # ------------------------------------------------------------------
    # STEP 1: Clusterer check + classifier expansion  (Task 3 lite)
    # ------------------------------------------------------------------

    def check_clusterer_and_expand(self, clusterer_labels, ground_truth_labels, classifier):
        clusterer_labels = np.array(clusterer_labels)
        valid_clusters = clusterer_labels[clusterer_labels != -1]
        unique_clusters = np.unique(valid_clusters)
        new_class_detected = len(unique_clusters) > 0

        if new_class_detected:
            print(f"✓ Clusterer detected new cluster(s). Expanding classifier by 1.")
            classifier.expand_classifier(num_to_add=1)
            return 1
        else:
            print("⚠ Clusterer did NOT detect any new class.")
            if self.use_intervention_override:
                print(f"  → [INTERVENTION ON] Forcing classifier expansion by 1.")
                classifier.expand_classifier(num_to_add=1)
                return 1
            else:
                print("  → [INTERVENTION OFF] Leaving classifier unchanged.")
                return 0

    # ------------------------------------------------------------------
    # STEP 2: Recording results into the matrix
    # ------------------------------------------------------------------

    def record_accuracy(self, task_index, accuracy):
        """
        Record the accuracy score for a specific past task at the current step.

        Call this once per past task after each training phase.

        Args:
            task_index (int): Which task's test split we just evaluated (0-indexed).
            accuracy (float): Accuracy score (0.0 to 1.0).
        """
        if task_index > self.current_step:
            raise ValueError(
                f"task_index ({task_index}) can't be greater than current_step ({self.current_step}). "
                f"You can only evaluate tasks you've already trained on."
            )
        self.performance_matrix[self.current_step, task_index] = accuracy

    # ------------------------------------------------------------------
    # STEP 3: Advancing to the next task
    # ------------------------------------------------------------------

    def advance_to_next_task(self):
        """
        Call this after you've finished training AND evaluating all test splits
        for the current step. It saves a snapshot and moves the step counter forward.
        """
        # Save a snapshot of metrics for this step
        snapshot = {
            "step": self.current_step,
            "avg_accuracy": self.calculate_average_accuracy(),
            "forgetting": self.calculate_forgetting_measure(),
            "row": self.performance_matrix[self.current_step, :self.current_step + 1].tolist(),
        }
        self.history.append(snapshot)

        print(f"\n--- Step {self.current_step} complete ---")
        print(f"    Avg Accuracy (this step): {snapshot['avg_accuracy']:.4f}")
        print(f"    Forgetting Measure:       {snapshot['forgetting']:.4f}")
        print(f"    Row scores: {[f'{x:.3f}' for x in snapshot['row']]}")

        self.current_step += 1
        print(f"\n=== Now starting Step {self.current_step} ===\n")

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def calculate_average_accuracy(self):
        """
        Average Accuracy (AA) — measures how well the model currently performs
        across all tasks it has been trained on so far. Higher = better plasticity.

        Formula: mean of performance_matrix[current_step, 0..current_step]
        """
        row = self.performance_matrix[self.current_step, :self.current_step + 1]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            return 0.0
        return float(np.mean(valid))

    def calculate_forgetting_measure(self):
        """
        Forgetting Measure (FM) — measures how much performance on old tasks has
        degraded since they were first learned. Higher = more catastrophic forgetting.

        Formula: for each past task j, take (peak accuracy ever achieved on j)
                 minus (current accuracy on j), then average across all past tasks.

        Returns 0.0 if we're still on the first task (nothing to forget yet).
        """
        if self.current_step < 1:
            return 0.0

        forgetting_scores = []
        for j in range(self.current_step):   # for each past task
            col = self.performance_matrix[:self.current_step + 1, j]
            valid = col[~np.isnan(col)]
            if len(valid) < 2:
                continue
            peak = float(np.max(valid))
            current = float(self.performance_matrix[self.current_step, j])
            if not np.isnan(current):
                forgetting_scores.append(peak - current)

        if not forgetting_scores:
            return 0.0
        return float(np.mean(forgetting_scores))

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def print_matrix(self):
        """Pretty-print the full performance matrix."""
        print("\n=== Performance Matrix ===")
        print(f"Rows = learning step, Columns = task being evaluated")
        print(f"(nan = not yet evaluated)\n")

        header = "       " + "  ".join([f"Task{j}" for j in range(self.num_tasks)])
        print(header)
        for i in range(self.num_tasks):
            row_vals = []
            for j in range(self.num_tasks):
                val = self.performance_matrix[i, j]
                row_vals.append(f"{val:.3f}" if not np.isnan(val) else "  nan")
            print(f"Step {i}: " + "  ".join(row_vals))
        print()

    def summary(self):
        """Print a full summary of all recorded steps."""
        print("\n=== Experiment Summary ===")
        for snap in self.history:
            print(f"  Step {snap['step']}: AA={snap['avg_accuracy']:.4f}, "
                  f"FM={snap['forgetting']:.4f}, "
                  f"Scores={[f'{x:.3f}' for x in snap['row']]}")
        self.print_matrix()
