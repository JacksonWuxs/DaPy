def edit_distance(lstr, rstr):
    assert hasattr(lstr, '__getitem__')
    assert hasattr(rstr, '__getitem__')
    len_l, len_r = len(lstr), len(rstr)
    matrix = [[x + y for y in xrange(len_r + 1)] for x in xrange(len_l + 1)]

    for i in xrange(1, len_l + 1):
        for j in xrange(1, len_r + 1):
            if lstr[i - 1] == rstr[j - 1]:
                d = 0
            else:
                d = 2
            matrix[i][j] = min(matrix[i-1][j] + 1,
                               matrix[i][j-1] + 1,
                               matrix[i-1][j-1] + d)
    return matrix[len_l][len_r]

def edit_ratio(lstr, rstr):
    lenth = len(lstr) + len(rstr)
    return (lenth - edit_distance(lstr, rstr)) / float(lenth)

if __name__ == '__main__':
    from pprint import pprint
    pprint(edit_distance('Hello World', 'HalloWorld'))
    pprint(edit_distance('HelloWorld', 'Hallo World'))
    pprint(edit_ratio('Hello World', 'HalloWorld'))
