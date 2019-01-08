import pandas as pd
data = [[1 ,95001, 1 ,92, '+'],[2 ,95001, 2 ,85, '-'],[3 ,95001, 3 ,88, '+'],[4 ,95002, 2 ,90, '+'],[5 ,95002, 3 ,80, '-']]
SC = pd.DataFrame(data,columns=['SC_ID', '  Sno','Cno','Grade','Class'])
print('R1')
print(SC)
print()
data = [[1 ,'Database', 4],[2 ,'PASCAL ',2],[3 ,'DataStructure', 4]]
COURSE =pd.DataFrame(data,columns=['Cno',' Cname',' Credit'])
print()
print('R2')
print((COURSE))
print()
result = pd.merge(SC, COURSE, on='Cno')
print()
print('NORMAL JOIN')
print((result))
print()
#r1=result[result['Cno'].isin([2])]
#print(r1.iloc[:,[0,2,4]])
r1=result.groupby(['Cno']).agg(lambda x: tuple(x)).applymap(list).reset_index()
print()
print('TUPLE ID PROPOGATION')
print(r1.iloc[:,[0,1,4]])

