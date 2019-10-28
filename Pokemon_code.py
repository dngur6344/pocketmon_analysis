import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
import csv
from pandas import Series, DataFrame

new_total = []
num_High = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
num_Mid = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
num_Low = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
num_lgd_High = 0
num_lgd_Mid = 0
num_lgd_Low = 0
#num_nolgd_High = 0
#num_nolgd_Mid = 0
#num_nolgd_Low = 0
num_lgd = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
strong_pok=[]
weak_pok=[]
sw_pair={'HP':0,'Attack':0,'Defense':0,'Sp. Atk':0,'Sp. Def':0,'Speed':0}
ws_pair={'HP':0,'Attack':0,'Defense':0,'Sp. Atk':0,'Sp. Def':0,'Speed':0}
sw_100_pair={'HP':0,'Attack':0,'Defense':0,'Sp. Atk':0,'Sp. Def':0,'Speed':0}
ws_100_pair={'HP':0,'Attack':0,'Defense':0,'Sp. Atk':0,'Sp. Def':0,'Speed':0}
strong_tot=[]
weak_tot=[]
feature=['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']

data = pd.read_csv('Pokemon.csv')

data_rating=data.pivot_table('Total',index='HP',columns='Generation',aggfunc='mean')
data_rating.head()

for i in range(800):
    new_sum = (int(data['HP'][i]) / 255 * 17) + (int(data["Attack"][i]) / 190 * 17) \
              + (int(data["Defense"][i]) / 230 * 17) + (int(data["Sp. Atk"][i]) / 194 * 16) \
              + (int(data["Sp. Def"][i]) / 230 * 16) + (int(data["Speed"][i]) / 180 * 17)
    new_total.append(int(new_sum))

    if new_sum >= 49:
        num_High[data['Generation'][i]] = num_High[data['Generation'][i]] + 1
        if data['Legendary'][i] == True:
            num_lgd_High = num_lgd_High + 1
#        elif data['Legendary'][i] == False:
#            num_nolgd_High=num_nolgd_High+1
        strong_pok.append(data['Name'][i])
    elif new_sum <= 19:
        num_Low[data['Generation'][i]] = num_Low[data['Generation'][i]] + 1
        if data['Legendary'][i] == True:
            num_lgd_Low = num_lgd_Low + 1
#        elif data['Legendary'][i] == False:
#            num_nolgd_Low=num_nolgd_Low+1
        weak_pok.append(data['Name'][i])
    else:
        num_Mid[data['Generation'][i]] = num_Mid[data['Generation'][i]] + 1
        if data['Legendary'][i] == True:
            num_lgd_Mid = num_lgd_Mid + 1
#        elif data['Legendary'][i] == False:
#            num_nolgd_Mid=num_nolgd_Mid+1
    if data['Legendary'][i] == True:
        num_lgd[data['Generation'][i]] = num_lgd[data['Generation'][i]] + 1

data['New_Total']=new_total

high_pok=num_High[1]+num_High[2]+num_High[3]+num_High[4]+num_High[5]+num_High[6]
mid_pok=num_Mid[1]+num_Mid[2]+num_Mid[3]+num_Mid[4]+num_Mid[5]+num_Mid[6]
low_pok=num_Low[1]+num_Low[2]+num_Low[3]+num_Low[4]+num_Low[5]+num_Low[6]
xs=["High","Middle","Low"]
ys=[high_pok,mid_pok,low_pok]
plt.bar(xs, ys)
plt.xlabel("Rank")
plt.ylabel("Number of Poketmon")
plt.show()


#2번 그래프 (total별 포켓몬 수)
sns.countplot(x=new_total, data=data)
plt.title('distribution of the poketmon by total score')
plt.xlabel('total score')
plt.ylabel('number of poketmon')
plt.show()

#3번 그래프 (세대별 상 중 하 포켓몬 비율)
num_High = pd.Series(num_High)
num_Mid = pd.Series(num_Mid)
num_Low = pd.Series(num_Low)
#상 그래프
plt.pie(num_High, labels=num_High.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of High rank by Generation')
plt.show()
#중 그래프
plt.pie(num_Mid, labels=num_Mid.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of Middle rank by Generation')
plt.show()
#하 그래프
plt.pie(num_Low, labels=num_Low.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of Low rank by Generation')
plt.show()

#4번 그래프 (전설 포켓몬 상 중 하 비율)
num_lgd = pd.Series(num_lgd)
xs = ["High", "Middle", "Low"]
ys = [num_lgd_High, num_lgd_Mid, num_lgd_Low]
#레전드 포켓몬의 세대별 수
plt.pie(num_lgd, labels=num_lgd.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of Legendary Poketmon by Generation')
plt.show()
#레전드 포켓몬의 상중하 랭크의 비율
plt.bar(xs, ys)
plt.xlabel("Rank")
plt.ylabel("Number of Poketmon")
plt.show()

str_idx=-1
wk_idx=-1
for index1 in strong_pok:
    for index2 in weak_pok:
        for i in range(800):
            if data['Name'][i]==index1:
                str_idx=i
            elif data['Name'][i]==index2:
                wk_idx=i
            if str_idx!=-1 and wk_idx!=-1:
                break
        if data['HP'][str_idx]>data['HP'][wk_idx]:
            sw_pair['HP']=sw_pair['HP']+1
        else:
            ws_pair['HP']=ws_pair['HP']+1
        if data['Attack'][str_idx]>data['Attack'][wk_idx]:
            sw_pair['Attack']=sw_pair['Attack']+1
        else:
            ws_pair['Attack']=ws_pair['Attack']+1
        if data['Defense'][str_idx]>data['Defense'][wk_idx]:
            sw_pair['Defense']=sw_pair['Defense']+1
        else:
            ws_pair['Defense']=ws_pair['Defense']+1
        if data['Sp. Atk'][str_idx]>data['Sp. Atk'][wk_idx]:
            sw_pair['Sp. Atk']=sw_pair['Sp. Atk']+1
        else:
            ws_pair['Sp. Atk']=ws_pair['Sp. Atk']+1
        if data['Sp. Def'][str_idx]>data['Sp. Def'][wk_idx]:
            sw_pair['Sp. Def'] = sw_pair['Sp. Def'] + 1
        else:
            ws_pair['Sp. Def'] = ws_pair['Sp. Def'] + 1
        if data['Speed'][str_idx]>data['Speed'][wk_idx]:
            sw_pair['Speed']=sw_pair['Speed']+1
        else:
            ws_pair['Speed']=ws_pair['Speed']+1
        strong_sum=(int(data['HP'][str_idx]) / 255 * 17) + (int(data["Attack"][str_idx]) / 190 * 17) \
              + (int(data["Defense"][str_idx]) / 230 * 17) + (int(data["Sp. Atk"][str_idx]) / 194 * 16) \
              + (int(data["Sp. Def"][str_idx]) / 230 * 16) + (int(data["Speed"][str_idx]) / 180 * 17)
        weak_sum=(int(data['HP'][wk_idx]) / 255 * 17) + (int(data["Attack"][wk_idx]) / 190 * 17) \
              + (int(data["Defense"][wk_idx]) / 230 * 17) + (int(data["Sp. Atk"][wk_idx]) / 194 * 16) \
              + (int(data["Sp. Def"][wk_idx]) / 230 * 16) + (int(data["Speed"][wk_idx]) / 180 * 17)
        strong_tot.append(strong_sum)
        weak_tot.append(weak_sum)
        str_idx=-1
        wk_idx=-1

for i in feature:
    sw_100_pair[i]=int(sw_pair[i]/(sw_pair[i]+ws_pair[i])*100)
    ws_100_pair[i]=int(ws_pair[i]/(sw_pair[i]+ws_pair[i])*100)

pyo=DataFrame([sw_pair,ws_pair,sw_100_pair,ws_100_pair],index=['strong>weak','weak>strong','strong>weak(%)','weak>strong(%)'])
print(pyo)#각 feature별 표야....직접 계산한 표...

'''csv파일을 하나 더 만든 이유는 로지스틱 회귀모델을 사용할 때 csv파일에 있는 모든 데이터를 다 사용하더라고 근데 우리는
강한 포켓몬, 중간 포켓몬, 약한 포켓몬 세 부류 중 중간 포켓몬을 사용하지 않잖아? 그래서 열심히 리스트도 만들어보고 별 짓
다 했는데도 안되길래 그냥 output.csv 파일을 만들고 그 안에 강한 포켓몬, 약한 포켓몬 다 넣자고 생각해서 만들게 되었어.'''

f=open('output.csv','w',newline='')
wr=csv.writer(f)
wr.writerow(['#','Name','Type 1','Type 2','Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Legendary','New_Total','flag'])
for i in range(800):
    if data['New_Total'][i]>=49:
        wr.writerow([i,data['Name'][i],data['Type 1'][i],data['Type 2'][i],data['Total'][i],data['HP'][i]\
                     ,data['Attack'][i],data['Defense'][i],data['Sp. Atk'][i],data['Sp. Def'][i],data['Speed'][i]\
                     ,data['Generation'][i],data['Legendary'][i],data['New_Total'][i],1.0])
    elif data['New_Total'][i]<=19:
        wr.writerow([i,data['Name'][i],data['Type 1'][i],data['Type 2'][i],data['Total'][i],data['HP'][i]\
                        ,data['Attack'][i],data['Defense'][i],data['Sp. Atk'][i],data['Sp. Def'][i],data['Speed'][i] \
                        , data['Generation'][i], data['Legendary'][i], data['New_Total'][i], 0.0])

#강한 포켓몬은 flag를 1로 해놓았고, 약한 포켓몬은 flag를 0으로 해놓았다.

f.close()
data2 = pd.read_csv('output.csv')

train_cols=data2.columns[5:11]
logit=sm.Logit(data2['flag'],data2[train_cols])
result=logit.fit()
print(result.summary())
'''여기서 주목할 건 coef(편회귀계수)라는 거래. 이 값이 양수이면 그 column의 값이 커질수록 flag가 1일 확률이
높아지고 반대로 값이 음수이면 그 column의 값이 커질수록 flag가 0일 확률이 높아진대.
그럼 여기서 HP의 coef가 -0.0495이므로 음수야. 그럼 HP가 높을수록 약한 포켓몬일 가능성(flag==0)이 높대....(왜지?)
암튼 그리고 Attack의 coef가 0.0415므로 양수야. 그럼 Attack이 높을수록 강한 포켓몬(flag==1)일 가능성이 높대!'''



print(np.exp(result.params))#오즈 비 구하는 것(성공할 확률이 실패할 확률보다 몇배가 높은가?)

'''Odds Ratio란 Odds의 비율이다. Odds란 성공/실패와 같이 상호 배타적이며 전체를 이루고 있는 것들의 비율을 의미한다. 
예를 들어 남자 승객의 경우 577명중 109명이 생존했다. 이 경우 Odds = P(생존)/P(사망) = (109/577)/(468/577) = 0.19/0.81 = 0.23
여자 승객의 경우 314명중 233명이 생존했다. 이 경우 Odds = P(생존)/P(사망) = (233/314)/(81/314) = 2.87
따라서 Odds Ratio = 0.23/2.87 = 약 0.08'''


data2['predict']=result.predict(data[train_cols])
print(data2.head())

'''참고 : http://3months.tistory.com/28?category=753896'''
#세대별로, 각 type 포켓몬의 수를 꺾은선 그래프로 표현한 것.
a = data.groupby(['Generation','Type 1']).count().reset_index()
a = a[['Generation', 'Type 1', 'New_Total']]
a = a.pivot('Generation', 'Type 1', 'New_Total')
a[['Water', 'Fire', 'Grass', 'Dragon', 'Normal', 'Rock', 'Flying', 'Electric']].plot(color=['b', 'r', 'g', '#FFA500', 'brown', '#6666ff', '#001012', 'y'], marker='o')
fig=plt.gcf()
fig.set_size_inches(12, 6)
plt.show()

#heatmap, 말 그대로 더 높은 수치일수록 뜨거워지는 표. 속성들간의 상관관계를 알기 쉬움
plt.figure(figsize=(10,6)) #manage the size of the plot
sns.heatmap(data.corr(), annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap
plt.show()

#type1별로 가장 강한 포켓몬
strong = data.sort_values(by='New_Total', ascending=False)
strong.drop_duplicates(subset=['Type 1'], keep='first')
print(strong)

#fire와 water의 공격력 방어렵 비교
fire = data[(data['Type 1'] == 'Fire') | ((data['Type 2']) == "Fire")]
water = data[(data['Type 1'] == 'Water') | ((data['Type 2']) == "Water")]
plt.scatter(fire.Attack.head(50), fire.Defense.head(50), color='R', label='Fire', marker="*", s=50)
plt.scatter(water.Attack.head(50), water.Defense.head(50), color='B', label="Water", s=25)
plt.xlabel("Attack")
plt.ylabel("Defense")
plt.legend()
plt.plot()
fig = plt.gcf()  #get the current figure using .gcf()
fig.set_size_inches(12, 6) #set the size for the figure
plt.show()

#legend 포켓몬이 다른 포켓몬에 비해 얼마나 강한 스탯을 갖고 있는지(New_Total 기준, 이것도 다른 속성 변수로 대체 가능함) 한눈에 파악 가능
plt.figure(figsize=(12, 6))
top_types = data['Type 1'].value_counts()[:10] #take the top 10 Types
df1 = data[data['Type 1'].isin(top_types.index)] #take the pokemons of the type with highest numbers, top 10
sns.swarmplot(x='Type 1', y='New_Total', data=df1, hue='Legendary') # this plot shows the points belonging to individual pokemons
# It is distributed by Type
plt.axhline(df1['New_Total'].mean(), color='red', linestyle='dashed')
plt.show()

#type별 Attack 분포도, 원하는 속성으로 교체 가능(y 변수 대입값만 바꿔주면 됨, x도 마찬가지)
plt.subplots(figsize=(15, 5))
plt.title('Attack by Type 1')
sns.boxplot(x="Type 1", y="Attack", data=data)
plt.ylim(0, 200)
plt.show()

#type별 포켓몬 분포도
labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'
sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]
colors = ['Y', 'B', '#00ff00', 'C', 'R', 'G', 'silver', 'white', 'M']
explode = (0.1, 0, 0.1, 0, 0., 0, 0, 0, 0)  # only "explode" the 3rd slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title("Percentage of Different Types of Pokemon")
plt.plot()
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.show()
