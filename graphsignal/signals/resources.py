import logging
import threading
import time

logger = logging.getLogger('graphsignal')

MAX_TAG_KEY_LEN = 50
MAX_TAG_VALUE_LEN = 250
MAX_ATTRIBUTE_VALUE_LEN = 2500

# Kinds whose attribute values are stored whole. The engine resource carries
# the engine's effective configuration as one JSON payload, and a cut would
# leave JSON that no longer parses. Tags stay capped: they are the resource's
# identity.
UNCAPPED_ATTRIBUTE_KINDS = frozenset({'engine'})


class ResourceStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._resources = {}

    def update_resource(
            self,
            kind,
            tags=None,
            attributes=None,
            first_seen_ts=None,
            last_seen_ts=None):
        if kind is None:
            return

        now_ns = time.time_ns()
        if first_seen_ts is None:
            first_seen_ts = now_ns
        if last_seen_ts is None:
            last_seen_ts = now_ns

        norm_tags = {}
        if tags:
            for key, value in tags.items():
                norm_tags[str(key)[:MAX_TAG_KEY_LEN]] = str(value)[:MAX_TAG_VALUE_LEN]

        value_cap = None if kind in UNCAPPED_ATTRIBUTE_KINDS else MAX_ATTRIBUTE_VALUE_LEN
        norm_attributes = {}
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                norm_attributes[str(key)[:MAX_TAG_KEY_LEN]] = str(value)[:value_cap]

        resource_key = (kind, frozenset(norm_tags.items()))

        with self._lock:
            existing = self._resources.get(resource_key)
            if existing is not None:
                first_seen_ts = min(existing['first_seen_ts'], first_seen_ts)
                last_seen_ts = max(existing['last_seen_ts'], last_seen_ts)
            self._resources[resource_key] = {
                'kind': kind,
                'tags': norm_tags,
                'attributes': norm_attributes,
                'first_seen_ts': first_seen_ts,
                'last_seen_ts': last_seen_ts,
            }

    def export(self):
        """Snapshot of all resources; never resets the store."""
        with self._lock:
            return [
                {**resource,
                 'tags': dict(resource['tags']),
                 'attributes': dict(resource['attributes'])}
                for resource in self._resources.values()]

    def clear(self):
        with self._lock:
            self._resources.clear()
