"""Sensor capabilities from the bundled SDK's ESimOne_Node_Type enum."""

TARGET_INGESTION_VERSION = "sensor-discovery-v4"

# Kinds whose objects are read through the target detection API.
DETECTION_KINDS = ("camera", "lidar", "perfect", "fusion")

# Every kind this module may report. A configuration override may only map a
# raw platform string onto one of these; it cannot invent a new kind.
KNOWN_KINDS = DETECTION_KINDS + ("radar", "ultrasonic", "gnss", "v2x")


def sensor_kind(value, type_map=None):
    """Accept SDK numeric strings and platform names without guessing from IDs.

    `type_map` lets a deployed platform name a detection sensor differently
    without a code change. It is matched before the built-in table, so a site
    can correct a wrong guess (for example a platform whose "6" is a camera
    rather than GNSS) without forking this module.
    """
    if isinstance(value, bytes):
        value = value.split(b"\0", 1)[0].decode("utf-8", "replace")
    raw = str(getattr(value, "value", value)).strip()
    if type_map:
        mapped = type_map.get(raw.lower())
        if mapped is not None:
            return mapped
    text = raw.lower()
    numeric = {"1": "camera", "2": "lidar", "3": "radar",
               "4": "ultrasonic", "5": "ultrasonic", "6": "gnss",
               "7": "perfect", "8": "v2x", "9": "fusion"}
    if text in numeric:
        return numeric[text]
    for word in ("ultrasonic", "perfect", "fusion", "camera", "lidar", "radar"):
        if word in text:
            return word
    return "unknown"


def parse_type_map(entries):
    """Validate `raw=kind` overrides into a lowercase lookup.

    Rejects rather than ignores: a mistyped override that silently did nothing
    would look exactly like the mapping not being needed, which is the
    confusion this whole report exists to remove.
    """
    if entries is None:
        return {}
    if isinstance(entries, dict):
        items = entries.items()
    elif isinstance(entries, str):
        items = []
        for chunk in entries.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError("sensor type map entry {0!r} is not raw=kind".format(chunk))
            raw, kind = chunk.split("=", 1)
            items.append((raw, kind))
    else:
        raise ValueError("sensor type map must be a mapping or comma separated string")
    result = {}
    for raw, kind in items:
        name = str(raw).strip().lower()
        target = str(kind).strip().lower()
        if not name:
            raise ValueError("sensor type map has an empty raw type")
        if target not in KNOWN_KINDS:
            raise ValueError("sensor type map {0}={1!r} is not one of {2}".format(
                name, target, "/".join(KNOWN_KINDS)))
        result[name] = target
    return result


def target_sensor_ids(configurations, preferred="", type_map=None):
    """Configured detection sources only; GNSS/radar/ultrasound use other APIs."""
    ids = []
    for item in configurations:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id", "")
        if (isinstance(identifier, str) and identifier
                and sensor_kind(item.get("type", ""), type_map) in DETECTION_KINDS
                and identifier not in ids):
            ids.append(identifier)
    if preferred in ids:
        ids.remove(preferred)
        ids.insert(0, preferred)
    return ids


def sensor_catalog_report(configurations, type_map=None):
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
            kind = sensor_kind(raw_type, type_map)
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
        "type_map": dict(type_map) if type_map else {},
    }
