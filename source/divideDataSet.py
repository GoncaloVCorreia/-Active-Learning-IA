import csv




#total_data_file=open("ALL_Datasets/Youtube01-Psy.csv","r")
total_data_file=open("ALL_Datasets/Youtube05-Shakira.csv","r")




total_data=csv.reader(total_data_file)

numSpam=0
numTotal=0
for row in total_data:
    numTotal+=1
    if(row[4]=="1"):
        numSpam+=1
numTotal=numTotal-1
print(numSpam)
print(numTotal)

portion=round(numTotal*0.85)
print(portion)


limit= round(portion/2)
print("Numero de Spam instances para treino: " +str(limit))
print("Numero de HAM instances para treino: " +str(limit))
print("Numero de SPAM instances para test: " +str(numSpam- limit))
print("Numero de HAM instances para test: " +str(numTotal-numSpam -limit))



i=0
j=0

"""
total_data_file2=open("ALL_Datasets/Youtube01-Psy.csv","r")
train_PSY=open("data/train_PSY.csv","w")
test_PSY=open("data/test_PSY.csv","w")
"""
"""
total_data_file2=open("ALL_Datasets/Youtube02-KatyPerry.csv","r")
train_PSY=open("data/train_katy.csv","w")
test_PSY=open("data/test_katy.csv","w")
"""
"""
total_data_file2=open("ALL_Datasets/Youtube03-LMFAO.csv","r")
train_PSY=open("data/train_LMFAO.csv","w")
test_PSY=open("data/test_LMFAO.csv","w")
"""

"""
total_data_file2=open("ALL_Datasets/Youtube04-Eminem.csv","r")
train_PSY=open("data/train_em.csv","w")
test_PSY=open("data/test_em.csv","w")
"""


total_data_file2=open("ALL_Datasets/Youtube05-Shakira.csv","r")
train_PSY=open("data/train_shakira.csv","w")
test_PSY=open("data/test_shakira.csv","w")

total_data2=csv.reader(total_data_file2)
train_PSY.write("COMMENT_ID, AUTHOR, DATE, CONTENT, CLASS \n")
test_PSY.write("COMMENT_ID, AUTHOR, DATE, CONTENT, CLASS \n")
writer=csv.writer(train_PSY)
writer2=csv.writer(test_PSY)



for line in total_data2:
    if(i<=limit and line[4]=="1"):
        writer.writerow(line)
        i+=1

    if(i > limit and line[4]=="1"):
        writer2.writerow(line)   
       
    if(j<=limit and line[4]=="0"):
        writer.writerow(line)
        j=j+1
       
    if(j> limit and line[4]=="0"):
        writer2.writerow(line)

       
  
