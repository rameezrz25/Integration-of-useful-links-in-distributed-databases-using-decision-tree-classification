import pandas as pd
data = [[1 ,95001, 1 ,92, '+'],[2 ,95001, 2 ,85, '-'],[3 ,95001, 3 ,88, '+'],[4 ,95002, 2 ,90, '+'],[5 ,95002, 3 ,80, '-']]
SC = pd.DataFrame(data,columns=['SC_ID', '  Sno','Cno','Grade','Class'])
print(SC)
data = [[1 ,'Database', 4],[2 ,'PASCAL ',2],[3 ,'DataStructure', 4]]
COURSE =pd.DataFrame(data,columns=['Cno',' Cname',' Credit'])
print((COURSE))
#print(set(SC['Cno']).intersection(set(COURSE['Cno'])))
#mergedStuff = pd.merge(SC,COURSE, on=['Cno'], how='outer')
uk = SC['Cno'].isin(COURSE['Cno']).astype(int)
print(sum(uk))
