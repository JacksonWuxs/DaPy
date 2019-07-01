from ctypes import Structure, POINTER, pointer, c_int as C_INT, byref
from collections import Sequence, namedtuple
class intLinkedNode(Structure):
    pass

intLinkedNode._fields_ = [
    ('next_', POINTER(intLinkedNode)),
    ('val', C_INT),
    ]

##class LinkedArray(Structure):
##    pass
##
##LinkedArray._fields_ = [
##    ('root', POINTER(intLinkedNode)),
##    ('tail', POINTER(intLinkedNode)),
##    ('node', C_INT)
##    ]

class intLinkedNode(object):
    def __init__(self, next=None, val=None):
        self.next = next
        self.val = val

def _append_left(link, new_val):
    return intLinkedNode(val=new_val, next_=pointer(link))

def _show_values(link):
    current_node = link
    while bool(current_node):
        yield current_node.val
        try:
            current-node = current_node.next.contents
        except ValueError:
            break
    else:
        yield current_node.val

# CFUNCTYPE(restype, *argtypes, **kwrds)


        
##class LinkedArray(Sequence):
##    def __init__(self, iterable=None):
##        self.root = intLinkedNode(val=0)
##        self.tail = self.root
##        self.node = 0
##        
##        if iterable is not None:
##            for value in iterable:
##                self.append(value)
##
##    def append(self, data):
##        self.tail.next = intLinkedNode(val=data) # pointer(next_node)
##        self.tail = self.tail.next
##        self.node += 1
##
##    def __len__(self):
##        return self.node
##
##    def __getitem__(self, index):
##        assert isinstance(index, int)
##        for i, node in enumerate(self):
##            if i == index:
##                return node.val
##
##    def __iter__(self):
##        current_node = self.root.next# .contents
##        while bool(current_node):
##            yield current_node.val
##            try:
##                current_node = current_node.next# .contents
##            except ValueError:
##                break
##        else:
##            yield current_node.val

if __name__ == '__main__':
    from random import randint
    linked = LinkedArray()

    for i in range(10):
        linked.append(randint(10, 20))
        
    print 'Iter:', [val for val in linked]
    print 'Length:', len(linked)
