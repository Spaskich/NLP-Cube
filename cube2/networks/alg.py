class FlatMap():
    def __init__(self):
        self._keys = []
        self._objects = []

    def __setitem__(self, key, item):
        try:
            index = self._keys.index(key)
            self._objects[index] = item
        except:
            self._keys.append(key)
            self._objects.append(item)

    def __getitem__(self, key):
        index = self._keys.index(key)
        return self._objects[index]

    def __repr__(self):
        return ' '.join(['{0}:{1}'.format(k, v) for k, v in zip(self._keys[:min(len(self._keys, 15))], self._objects)])

    def __len__(self):
        return len(self._keys)

    def __delitem__(self, key):
        index = self._keys.index(key)
        self._keys = self._keys[:index] + self._keys[index + 1:]
        self._objects = self._objects[:index] + self._objects[index + 1:]

    def clear(self):
        self._keys = []
        self._objects = []
        return self

    def copy(self):
        new_inst = FlatMap()
        for k, v in zip(self._keys, self._objects):
            new_inst._keys.append(k)
            new_inst._objects.append(v)
        return new_inst

    def has_key(self, k):
        return k in self._keys

    # def update(self, *args, **kwargs):
    #     return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self._keys

    def values(self):
        return self._objects

    def items(self):
        return [(k, v) for k, v in zip(self._keys, self._objects)]

    # def pop(self, *args):
    #     return self.__dict__.pop(*args)

    # def __cmp__(self, dict_):
    #     return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        # keys should come sorted here
        start = 0
        end = len(self._keys) - 1
        while start < end:
            mid = (start + end) // 2
            if self._keys[mid] == item:
                return True
            if self._keys[mid] > item:
                end = mid - 1
            else:
                start = mid + 1

        return False

    def __iter__(self):
        return iter(self._keys)

    # def __unicode__(self):
    #     return unicode(repr(self.__dict__))
