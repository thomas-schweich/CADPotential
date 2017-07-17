import dxfgrabber

dxf = dxfgrabber.readfile('potential2000.dxf')

print ',\n'.join(dxf.entities[0].acis)


'''
types = [e.dxftype for e in dxf.entities]

unique = {e for e in types}

uniqueEntityNames = set()
uniqueEntities = []
for e in dxf.entities:
    if e.dxftype not in uniqueEntityNames:
        uniqueEntityNames.add(e.dxftype)
        uniqueEntities += [e]

print unique

print '\n'.join(['All attributes of each type:']
                + ['\n%s:\n%s'
                   % (str(t), str(t.__dict__))
                   for t in uniqueEntities])
'''