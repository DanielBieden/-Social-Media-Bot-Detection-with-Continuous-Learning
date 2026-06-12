class YAJLStreamSanitizer:
    """
    Intercepts bytes before ijson reads them. Replaces illegal raw 
    control characters with safe spaces using high-speed byte translation.
    """
    # Characters 0-31 are illegal in JSON strings, except \n (10) and \r (13)
    _control_bytes = bytes([i for i in range(32) if i not in (10, 13)])
    _safe_spaces = b' ' * len(_control_bytes)
    _trans_table = bytes.maketrans(_control_bytes, _safe_spaces)

    def __init__(self, file_obj):
        self.file_obj = file_obj

    def read(self, size=-1):
        chunk = self.file_obj.read(size)
        if not chunk:
            return chunk
        return chunk.translate(self._trans_table)