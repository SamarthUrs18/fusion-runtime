"""Turn detectors: deciding when the user has finished speaking.

Built in: `silence` (the default: the turn ends after a configurable pause).
Models plug in through the registry, as "module:Class" or an entry point
named "turn.<name>":

    [project.entry-points."fusion_runtime.runtimes"]
    "turn.my_detector" = "my_package.turns:MyDetector"

and is selected in config:

    PipelineConfig(turn_detection=TurnDetectionConfig(runtime="my_detector", model="...", options={...}))

See fusion_runtime.contract.turn for the interface and how the engine uses it.
"""
from fusion_runtime.contract import ModelSpec


def turn_detector_spec(config) -> ModelSpec:
    """The ModelSpec for a TurnDetectionConfig: runtime name, optional model reference, options."""
    runtime = config.runtime or getattr(config.provider, "value", None) or "silence"
    return ModelSpec(stage="turn", runtime=runtime, model=config.model or "", options=dict(config.options))
