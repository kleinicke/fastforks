# Vendored from pq-yang/MatAnyone e5ddc534c1fff9bb9e54cf476095d29071b7cb4f, matanyone/inference/object_info.py (S-Lab License 1.0, ../../LICENSE).
# fastforks: imports `matanyone.*` -> the vendored `fastforks.matting.models.matanyone.net.*`; other changes are marked `fastforks:`.
class ObjectInfo:
    """
    Store meta information for an object
    """
    def __init__(self, id: int):
        self.id = id
        self.poke_count = 0  # count number of detections missed

    def poke(self) -> None:
        self.poke_count += 1

    def unpoke(self) -> None:
        self.poke_count = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if type(other) == int:
            return self.id == other
        return self.id == other.id

    def __repr__(self):
        return f'(ID: {self.id})'
