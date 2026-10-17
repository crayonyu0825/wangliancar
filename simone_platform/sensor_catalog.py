"""Sensor capabilities from the bundled SDK's ESimOne_Node_Type enum."""

TARGET_INGESTION_VERSION = "sensor-discovery-v3"

# Kinds whose objects are read through the target detection API.
DETECTION_KINDS = ("camera", "lidar", "perfect", "fusion")


def sensor_kind(value):
    """Accept SDK numeric strings and platform names without guessing from IDs."""
    if isinstance(value, bytes):
        value = value.split(b"\0", 1)[0].decode("utf-8", "replace")
    value = str(getattr(value, "value", value)).strip().lower()
    numeric = {"1": "camera", "2": "lidar", "3": "radar",
               "4": "ultrasonic", "5": "ultrasonic", "6": "gnss",
               "7": "perfect", "8": "v2x", "9": "fusion"}
    if value in numeric:
        return numeric[value]
    for word in ("ultrasonic", "perfect", "fusion", "camera", "lidar", "radar"):
        if word in value:
            return word
    return "unknown"


def target_sensor_ids(configurations, preferred=""):
    """Configured detection sources only; GNSS/radar/ultrasound use other APIs."""
    ids = []
    for item in configurations:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id", "")
        if (isinstance(identifier, str) and identifier
                and sensor_kind(item.get("type", "")) in DETECTION_KINDS
                and identifier not in ids):
            ids.append(identifier)
    if preferred in ids:
        ids.remove(preferred)
        ids.insert(0, preferred)
    return ids


def sensor_catalog_report(configurations):
    """Describe configured sensors so an unrecognised type cannot hide.

    `target_sensor_ids` only returns sources whose type is understood, which is
    the safe reading rule. The consequence is that a detection sensor published
    under a type name this binding does not know looks identical to a vehicle
    with no detection sensors at all. This report keeps that distinction
    visible: it names every sensor, its resolved kind, and the raw type value.

    The reading rule is unchanged; nothing here makes an unknown type a target
    source. It only stops a platform-side naming difference from being reported
    as "not configured".
    """
    sensors = []
    counts = {}
    if isinstance(configurations, (list, tuple)):
        for item in configurations:
            if not isinstance(item, dict):
                continue
            raw_type = item.get("type", "")
            kind = sensor_kind(raw_type)
            counts[kind] = counts.get(kind, 0) + 1
            sensors.append({
                "id": item.get("id", ""),
                "kind": kind,
                "raw_type": raw_type if isinstance(raw_type, str) else repr(raw_type),
                "detection": kind in DETECTION_KINDS,
            })
    detection = [item["id"] for item in sensors if item["detection"]]
    unclassified = [item for item in sensors if item["kind"] == "unknown"]
    if detection:
        status = "detection_source_configured"
    elif unclassified:
        # The likely cause of a target-scene zero: sensors exist, but none is
        # recognised as a detection source.
        status = "unclassified_sensor_types_only"
    elif sensors:
        status = "non_detection_sensors_only"
    else:
        status = "no_sensors_published"
    return {
        "status": status,
        "count": len(sensors),
        "counts_by_kind": counts,
        "detection_ids": detection,
        "unclassified": unclassified,
        "sensors": sensors,
    }
