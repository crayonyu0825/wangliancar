"""Shared SDK-free observation evidence; legacy decision import is preserved."""
from core.behavior_contract import TaskContext,finite,require


class Evidence(object):
    def __init__(self, context, frame_id, observed_at_s, valid_until_s, usable=False,
                 coverage_verified=False, source="unknown", clock_id="process_monotonic",
                 source_kind="unknown"):
        require(isinstance(context, TaskContext) and type(frame_id) is int and frame_id >= 0,
                "invalid observation identity")
        require(finite(observed_at_s) and finite(valid_until_s) and type(usable) is bool
                and type(coverage_verified) is bool and isinstance(source, str), "invalid observation evidence")
        self.context, self.frame_id, self.observed_at_s, self.valid_until_s = context, frame_id, observed_at_s, valid_until_s
        require(source_kind in ("unknown", "sensor", "map", "task", "planning", "verified_fusion",
                                "synthetic", "diagnostic_ground_truth"), "invalid observation source kind")
        self.usable, self.coverage_verified, self.source, self.clock_id = usable, coverage_verified, source, clock_id
        self.source_kind = source_kind

    def verified(self, frame, max_age_s=0.5,
                 allowed_source_kinds=("sensor", "map", "task", "planning", "verified_fusion", "synthetic")):
        return (self.source_kind in allowed_source_kinds and self.usable and self.coverage_verified and self.source != "unknown"
                and self.context.key() == frame.context.key() and self.clock_id == frame.clock_id
                and self.frame_id <= frame.frame_id
                and self.observed_at_s <= frame.observed_at_s < self.valid_until_s
                and frame.observed_at_s - self.observed_at_s <= max_age_s
                and frame.current(frame.observed_at_s) and not frame.paused)
