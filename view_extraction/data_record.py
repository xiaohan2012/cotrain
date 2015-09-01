import sys

class DataRecord(object):
    def __init__(self, tokens, y=None, pos=None, const_parse=None, dep_parse=None):
        self.tokens = tokens
        self.pos = pos
        self.const_parse = const_parse
        self.dep_parse = dep_parse
        self._y = y
        
    @property
    def y(self):
        if self._y == None:
            sys.stderr.write("Warning: DataRecord.y is None")
        return self._y
