"""Fixed decision entry. Replace only the engine behind decide()."""

from members.decision.engine import DecisionEngine

DECISION_VERSION = "decision-main-pr23-integration-v1"
_ENGINE = DecisionEngine()


def reset_decision(settings=None,parking_inputs_provider=None,lane_change_inputs_provider=None,
                   driving_inputs_provider=None, task_inputs_provider=None):
    """Start a new runtime session without carrying prior blind-stop state."""
    global _ENGINE
    _ENGINE = DecisionEngine(settings,parking_inputs_provider=parking_inputs_provider,
                            lane_change_inputs_provider=lane_change_inputs_provider,
                            driving_inputs_provider=driving_inputs_provider,
                            task_inputs_provider=task_inputs_provider)


def decision_info():
    """Expose the running implementation and tuning for launcher diagnostics."""
    info = {"version": DECISION_VERSION, "engine": type(_ENGINE).__name__}
    info.update(vars(_ENGINE.settings))
    return info


def decision_settings():
    """Expose validated session settings to the captain's independent monitor."""
    return _ENGINE.settings


def decision_behavior_info():
    """Diagnostic copy only; execution feedback uses the public channel."""
    return _ENGINE.behavior_diagnostics()


def decide(perception):
    """Choose a behaviour for one perception frame.

    Input:  Perception (captain)
    Output: DecisionTarget (planning member)

    The engine is module-level so its anti-flapping state survives between
    frames. See members/decision/BASELINE.md for the behaviour table and the
    parameters that are still provisional.
    """
    return _ENGINE.run(perception)
