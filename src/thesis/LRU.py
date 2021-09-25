from collections import OrderedDict


class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        while sum(self.cache.values()) > self.capacity:
            self.cache.popitem(last=False)


if __name__ == '__main__':
    cache = LRUCache(10)
    cache.put(1, 2)
    cache.put(1, 2)
    cache.put(1, 2)
    cache.put(1, 2)
    cache.put(1, 2)
    cache.put(1, 2)
    cache.put(1, 2)
