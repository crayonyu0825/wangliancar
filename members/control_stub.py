"""Fixed team entry for forward/reverse trajectory execution."""

from members.control.controller import ControlEngine
from members.control.parameters import VehicleCalibration


_engine = ControlEngine()


def configure_control(config):
    """Captain injects vehicle-specific calibration; never read SDK or INI here."""
    global _engine
    _engine = ControlEngine(VehicleCalibration.from_app_config(config))


def compute_control(perception, trajectory):
    output = _engine.compute(perception, trajectory)
    output.execution_feedback = _engine.execution_feedback(perception, trajectory, output)
    return output
