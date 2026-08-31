"""Driver Monitoring System — drowsiness, attention, and phone-usage detection.

The three detector classes are exported lazily (PEP 562). `from dms import
EarDetector` behaves exactly as before, but importing the package no longer
drags in OpenCV, MediaPipe and Ultralytics, so `import dms.logic` and
`import dms.config` cost nothing but the standard library. That is what lets
the unit tests run in CI without a two-gigabyte dependency install.
"""

__all__ = ["EarDetector", "EarConfig", "HeadPoseEstimator", "PoseConfig",
           "PhoneDetector", "PHONE_CLASS_ID"]

_EXPORTS = {
    "EarDetector": ".ear",
    "EarConfig": ".config",
    "HeadPoseEstimator": ".head_pose",
    "PoseConfig": ".config",
    "PhoneDetector": ".phone_detector",
    "PHONE_CLASS_ID": ".config",
}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
