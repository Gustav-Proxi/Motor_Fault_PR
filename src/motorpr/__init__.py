"""Bandwidth-limited pattern recognition on smartphone induction-motor vibration."""
import warnings

# numpy 2.x on Apple Accelerate raises spurious FP warnings from matmul even when the
# result is finite (reproducible with a plain random a.T @ a); silence only that message.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
