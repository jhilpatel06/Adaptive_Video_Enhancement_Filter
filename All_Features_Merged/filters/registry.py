class FilterRegistry:
    """
    A simple registry to keep track of all available filters.
    Makes it easy to switch filters via gestures, keyboard, or UI.
    """
    _filters = {}
    _metadata = {}  # Stores flags like is_face_filter, is_gesture_filter

    @classmethod
    def register(cls, name, func, is_face_filter=True, is_gesture_filter=False):
        """
        Registers a new filter function under the given name.

        Args:
            name:              Short unique name (e.g. "sepia", "gesture_temp").
            func:              The callable filter function.
            is_face_filter:    True if the filter needs per-face ROI processing.
            is_gesture_filter: True if the filter is driven by hand gesture input.
        """
        cls._filters[name] = func
        cls._metadata[name] = {
            "is_face_filter": is_face_filter,
            "is_gesture_filter": is_gesture_filter,
        }

    @classmethod
    def get(cls, name):
        """Retrieves a filter function by name."""
        return cls._filters.get(name)

    @classmethod
    def get_is_face_filter(cls, name):
        """Returns whether the filter requires face detection."""
        meta = cls._metadata.get(name, {})
        return meta.get("is_face_filter", True)

    @classmethod
    def get_is_gesture_filter(cls, name):
        """Returns whether the filter is gesture-driven."""
        meta = cls._metadata.get(name, {})
        return meta.get("is_gesture_filter", False)

    @classmethod
    def list_filters(cls):
        """Returns a list of all registered filter names."""
        return list(cls._filters.keys())
