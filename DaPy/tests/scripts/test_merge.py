import DaPy as dp

for key in (True, False, 'other', 'self'):
    for same in (True, False):
        left = dp.SeriesSet([
                        ['Alan', 35],
                        ['Bob', 27],
                        ['Charlie', 30],
                        ['Daniel', 29]],
                        ['Name', 'Age'])
        right = dp.SeriesSet([['Alan', 'M', 35],
                        ['Bob', 'M', 27],
                        ['Charlie', 'F', 30],
                        ['Janny', 'F', 26]],
                        ['Name', 'gender', 'Age'])
        
        print 'MERGE with keep_key=%s and keep_same=%s' % (key, same)
        left.merge(right, 'Name', 'Name', keep_key=key, keep_same=same)
        print left.show()
        print
data1 = dp.SeriesSet([['A', 39, 'F'], ['B', 40, 'F'], ['C', 38, 'M']],
                          ['Name', 'Age', 'Gender'])
data2 = dp.Frame([['A', 'F', True], ['B', 'F', False], ['C', 'M', True]],
                          ['Name', 'Gender', 'Married'])

data3 = [['A', 'China'], ['B', 'US'], ['C', 'Japan'], ['D', 'England']]
print dp.merge(data1, data2, data3, keys=0, keep_key='self', keep_same=False, ).show()
