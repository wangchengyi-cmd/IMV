#%%
import pandas as pd 
import numpy as np
import time
import concurrent
import gc
def missing_values_table(df):
    mis_val = df.isnull().sum() 
    mis_val_percent = 100 * df.isnull().sum() / len(df) 
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis= 1) 
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0:'Missing Values',
                                                              1:'% of Total Values'}) 
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values',ascending=False).round(1) 
    return mis_val_table_ren_columns 
def missing_values(i):
    data_t=data_test[data_test.icustay_id==i] 
    val_percent = round(100*data_t.isnull().sum().sum()/ data_t.size,1) 
    pp_val=pd.DataFrame(np.array([i,val_percent]).reshape(1,-1),columns=['icustay_id','missing_val']) 
    return pp_val 
data=pd.read_csv('final_dabiao.csv')
para_label=pd.read_csv('para_label.csv')
data['temperature'] = np.mean(pd.concat([data.temperature_bgart,data.temperature_bg,data.tempc],axis=1),axis=1) 
data.drop(['temperature_bgart','temperature_bg','tempc'],axis=1,inplace=True)
data['so2'] = np.mean(pd.concat([data.so2_bgart,data.so2_bg],axis=1),axis=1)
data.drop(['so2_bgart','so2_bg'],axis=1,inplace=True)
data['fio2'] = np.mean(pd.concat([data.fio2_bg,data.fio2_chartevents_bgart,data.fio2_bgart],axis=1),axis=1)
data.drop(['fio2_bg','fio2_chartevents_bgart','fio2_bgart'],axis=1,inplace=True)
data['baseexcess'] = np.mean(pd.concat([data.baseexcess_bg,data.baseexcess_bgart],axis=1),axis=1)
data.drop(['baseexcess_bg','baseexcess_bgart'],axis=1,inplace=True)
data['bicarbonate'] = np.mean(pd.concat([data.bicarbonate_bg,data.bicarbonate_bgart,data.bicarbonate_lab],axis=1),axis=1)
data.drop(['bicarbonate_bg','bicarbonate_bgart','bicarbonate_lab'],axis=1,inplace=True)
data['totalco2'] = np.mean(pd.concat([data.totalco2_bg,data.totalco2_bgart],axis=1),axis=1)
data.drop(['totalco2_bg','totalco2_bgart'],axis=1,inplace=True)
data['carboxyhemoglobin'] = np.mean(pd.concat([data.carboxyhemoglobin_bg,data.carboxyhemoglobin_bgart],axis=1),axis=1)
data.drop(['carboxyhemoglobin_bg','carboxyhemoglobin_bgart'],axis=1,inplace=True)
data['chloride'] = np.mean(pd.concat([data.chloride_bg,data.chloride_bgart,data.chloride_lab],axis=1),axis=1)
data.drop(['chloride_bg','chloride_bgart','chloride_lab'],axis=1,inplace=True)
data['calcium'] = np.mean(pd.concat([data.calcium_bg,data.calcium_bgart],axis=1),axis=1)
data.drop(['calcium_bg','calcium_bgart'],axis=1,inplace=True)
data['glucose'] = np.mean(pd.concat([data.glucose_bg,data.glucose_bgart,data.glucose_lab,data.glucose],axis=1),axis=1)
data.drop(['glucose_bg','glucose_bgart','glucose_lab'],axis=1,inplace=True)
data['hematocrit'] = np.mean(pd.concat([data.hematocrit_bg,data.hematocrit_bgart,data.hematocrit_lab],axis=1),axis=1)
data.drop(['hematocrit_bg','hematocrit_bgart','hematocrit_lab'],axis=1,inplace=True)
data['hemoglobin'] = np.mean(pd.concat([data.hemoglobin_bg,data.hemoglobin_bgart,data.hemoglobin_lab],axis=1),axis=1)
data.drop(['hemoglobin_bg','hemoglobin_bgart','hemoglobin_lab'],axis=1,inplace=True)
data['intubated'] = np.mean(pd.concat([data.intubated_bg,data.intubated_bgart],axis=1),axis=1)
data.drop(['intubated_bg','intubated_bgart'],axis=1,inplace=True)
data['lactate'] = np.mean(pd.concat([data.lactate_bg,data.lactate_bgart,data.lactate_lab],axis=1),axis=1)
data.drop(['lactate_bg','lactate_bgart','lactate_lab'],axis=1,inplace=True)
data['methemoglobin'] = np.mean(pd.concat([data.methemoglobin_bg,data.methemoglobin_bgart],axis=1),axis=1)
data.drop(['methemoglobin_bg','methemoglobin_bgart'],axis=1,inplace=True)
data['o2flow'] = np.mean(pd.concat([data.o2flow_bg,data.o2flow_bgart],axis=1),axis=1)
data.drop(['o2flow_bg','o2flow_bgart'],axis=1,inplace=True)
data['pco2'] = np.mean(pd.concat([data.pco2_bg,data.pco2_bgart],axis=1),axis=1)
data.drop(['pco2_bg','pco2_bgart'],axis=1,inplace=True)
data['peep'] = np.mean(pd.concat([data.peep_bg,data.peep_bgart],axis=1),axis=1)
data.drop(['peep_bg','peep_bgart'],axis=1,inplace=True)
data['ph'] = np.mean(pd.concat([data.ph_bg,data.ph_bgart],axis=1),axis=1)
data.drop(['ph_bg','ph_bgart'],axis=1,inplace=True)
data['po2'] = np.mean(pd.concat([data.po2_bg,data.po2_bgart],axis=1),axis=1)
data.drop(['po2_bg','po2_bgart'],axis=1,inplace=True)
data['requiredo2'] = np.mean(pd.concat([data.requiredo2_bg,data.requiredo2_bgart],axis=1),axis=1)
data.drop(['requiredo2_bg','requiredo2_bgart'],axis=1,inplace=True)
data['sodium'] = np.mean(pd.concat([data.sodium_bg,data.sodium_bgart,data.sodium_lab],axis=1),axis=1)
data.drop(['sodium_bg','sodium_bgart','sodium_lab'],axis=1,inplace=True)
data['potassium'] = np.mean(pd.concat([data.potassium_bg,data.potassium_bgart,data.potassium_lab],axis=1),axis=1)
data.drop(['potassium_bg','potassium_bgart','potassium_lab'],axis=1,inplace=True)
data['spo2'] = np.mean(pd.concat([data.spo2,data.spo2_bgart],axis=1),axis=1)
data.drop(['spo2_bgart'],axis=1,inplace=True)
data['tidalvolume'] = np.mean(pd.concat([data.tidalvolume_bgart,data.tidalvolume_bg],axis=1),axis=1)
data.drop(['tidalvolume_bgart','tidalvolume_bg'],axis=1,inplace=True)
data.drop(['ventilationrate_bg','ventilationrate_bgart'],axis=1,inplace=True)
data.drop(['ventilator_bg','ventilator_bgart'],axis=1,inplace=True)
data.drop(['aado2_bg','aado2_bgart','aado2_calc_bgart'],axis=1,inplace=True)
data.drop(['aniongap_lab'],axis=1,inplace=True)
data.drop(['endotrachflag','intubated'],axis=1,inplace=True)

todischarge=data['to_discharge']
data.drop(labels=['to_discharge'],axis=1,inplace=True)
data.insert(58,'to_discharge',todischarge)

deathlabel=data['death_label']
data.drop(labels=['death_label'],axis=1,inplace=True)
data.insert(57,'death_label',deathlabel)

ventstatus=data['vent_statu']
data.drop(labels=['vent_statu'],axis=1,inplace=True)
data.insert(56,'vent_statu',ventstatus)

map1=data['mean_air_pressure']
data.drop(labels=['mean_air_pressure'],axis=1,inplace=True)
data.insert(55,'mean_air_pressure',map1)

Charttime=data['charttime']
data.drop(labels=['charttime'],axis=1,inplace=True)
data.insert(2,'charttime',Charttime)

data.to_csv('/home/amms62037/wang/mimic/map_zhenghe_data.csv',index=False)

data=pd.read_csv('/home/amms62037/wang/mimic/map_zhenghe_data.csv') 
inr=[]
for i in range(101): 
    missing_df = missing_values_table(data) 
    missing_columns = list(missing_df[missing_df['% of Total Values'] > i].index)
    data_s = data.drop(columns = list(missing_columns)) 
    data_s = data_s.drop(columns = ['hadm_id', 'oxygentherapy_1', 'mechvent_1', 'vent_statu', 'mechvent_0', 'bmi', 'to_discharge', 'oxygentherapy_0', 'icustay_id', 'gender_0', 'charttime', 'gender_1', 'death_label','vent_statu','death_label']) # 删除无创参数，留下有创参数
    inr.append(para_label[list(data_s)].sum(axis=1)[0]/data_s.shape[1]) 
print(inr)  

icustay_id_set=(data.loc[:,['icustay_id','vent_statu']][data.loc[:,['icustay_id','vent_statu']].vent_statu<2].groupby(data.icustay_id,as_index=False)).max()

data_test=data[data.icustay_id.isin(icustay_id_set[icustay_id_set.vent_statu==1].icustay_id)]

with concurrent.futures.ProcessPoolExecutor() as executor:
    pro=list(executor.map(missing_values,set(data_test.icustay_id)))

for i in range(0,len(pro)):
    if i==0:
        input_mulit=('pro[{}]'.format(i))
    else:
        input_mulit=(input_mulit+',pro[{}]'.format(i))

p=pd.concat(eval(input_mulit),axis=0,ignore_index=True);comtest=p;


youchuangzhenghe_comtest=data[data.icustay_id.isin(comtest[comtest.missing_val<=65].icustay_id)] 

# #取无创的icustay_id(人数)
data_test=data[data.icustay_id.isin(icustay_id_set[icustay_id_set.vent_statu==0].icustay_id)]

with concurrent.futures.ProcessPoolExecutor() as executor:
    pro=list(executor.map(missing_values,set(data_test.icustay_id)))

for i in range(0,len(pro)):
    if i==0:
        input_mulit=('pro[{}]'.format(i))
    else:
        input_mulit=(input_mulit+',pro[{}]'.format(i))

p=pd.concat(eval(input_mulit),axis=0,ignore_index=True);comtest=p;
wuchuangzhenghe_comtest=data[data.icustay_id.isin(comtest[comtest.missing_val<=57].icustay_id)]
zhenghe_comtest=pd.concat([youchuangzhenghe_comtest,wuchuangzhenghe_comtest])
zhenghe_comtest.to_csv('map_del_patient_comtest.csv',index=False)
#%%
import numpy as np
import pandas as pd
import time 
import warnings

comtest = pd.read_csv("map_del_patient_comtest.csv") 
Charttime=comtest['charttime']
comtest.drop(labels=['charttime'],axis=1,inplace=True)
comtest.insert(5,'charttime',Charttime)
comtest.rename(columns={'charttime': 'chart_time'},inplace=True)
comtest.rename(columns={'pao2fio2ratio_bgart': 'pao2fio2ratio'},inplace=True)  
other_label=comtest.columns.values.tolist()

for h in ['icustay_id','hadm_id','age','gender_0','gender_1','mechvent_1','mechvent_0','oxygentherapy_1','oxygentherapy_0','urineoutput','fio2','pao2fio2ratio','peep']:
    print(h)
    other_label.remove(h)
def inter(i):
    icu_id_param=comtest[comtest.icustay_id==i]  #
    chart_time=list(set(icu_id_param.chart_time)) 
    chart_time.sort()
    # s=time.time() 
    times=0;
    for hour in chart_time:
        if np.isnan(np.array(icu_id_param.fio2.iloc[times])) and icu_id_param.mechvent_1.iloc[times]==0:  
            icu_id_param.fio2.iloc[times]=21
        else:
            if np.isnan(np.array(icu_id_param.fio2.iloc[times])) and icu_id_param.mechvent_1.iloc[times]==1: 
                icu_id_param.fio2.iloc[times]=icu_id_param.fio2.iloc[times-1]
    #对peep的插补
        if np.isnan(np.array(icu_id_param.peep.iloc[times])) and  icu_id_param.mechvent_1.iloc[times]== 0:
            icu_id_param.peep.iloc[times]=0
        elif np.isnan(np.array(icu_id_param.peep.iloc[times])) and icu_id_param.mechvent_1.iloc[times] == 1: 
            if times==0:
                icu_id_param.peep.iloc[times]=5
            else:
                if icu_id_param.mechvent_1.iloc[times-1] == 1:
                    icu_id_param.peep.iloc[times]=icu_id_param.peep.iloc[times-1]
                else:
                    icu_id_param.peep.iloc[times]=5
    #对urineoutput的插补
        if np.isnan(np.array(icu_id_param.urineoutput.iloc[times])):
            icu_id_param.urineoutput.iloc[times]=0
    #other_label
        for param in other_label:
            if np.isnan(np.array(icu_id_param[icu_id_param.chart_time==hour][param])) and times==0: 
                icu_id_param[param].iloc[times]=comtest[param].median()       
            else:
                if np.isnan(np.array(icu_id_param[icu_id_param.chart_time==hour][param])): 
                    icu_id_param[param].iloc[times]=icu_id_param[param].iloc[times-1]
        times+=1
    return icu_id_param   
import time
import warnings
from tqdm import tqdm 
warnings.filterwarnings("ignore")
s=time.time()
import concurrent
with concurrent.futures.ProcessPoolExecutor() as executor:
    pro=list(tqdm(executor.map(inter,set(comtest.icustay_id)),total=len(set(comtest.icustay_id)),desc='progress:')) 

for i in range(0,len(pro)):
    if i==0:
        input_mulit=('pro[{}]'.format(i))
    else:
        input_mulit=(input_mulit+',pro[{}]'.format(i))

p=pd.concat(eval(input_mulit),axis=0,ignore_index=True);comtest=p;
print(s-time.time())
###########################################osi计算方法#################################################################
osi = (comtest['mean_air_pressure'] * comtest['fio2'] )/comtest['spo2']
comtest.to_csv('map_comtest_1624_pat_intered.csv',index=False)
#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
comtest = pd.read_csv("/home/amms62037/wang/mimic/map_comtest_1624_pat_intered.csv")
learn_len=1;gap=4;fore_len=4 
old_name=['mechvent_1','mechvent_0','oxygentherapy_1','oxygentherapy_0','pao2fio2ratio', 'osi','gcs','gcsmotor','gcsverbal','gcseyes','albumin_lab','bands_lab','bilirubin_lab','creatinine_lab','platelet_lab','ptt_lab','inr_lab','pt_lab','bun_lab', 'wbc_lab','urineoutput','heartrate','sysbp','diasbp', 'meanbp','resprate','spo2','temperature','so2','fio2','baseexcess','bicarbonate','totalco2','carboxyhemoglobin','chloride','calcium','hematocrit','hemoglobin','lactate','methemoglobin','o2flow','pco2','peep','ph','po2','requiredo2','sodium','potassium','tidalvolume']
new_name=['icustay_id','hadm_id','gender_0','gender_1','age','bmi'] 
for j in range(0,len(old_name)):  
    for i in range(1,learn_len+1):
        new_name=new_name+[old_name[j]+'_'+str(i)] 
new_name=new_name+['vent_statu']+['death_label']+['to_discharge'] 
def inter_learn_len(chart_time_opt,icu_id_param,labelname):
    icu_id_param=icu_id_param[icu_id_param.chart_time<=chart_time_opt[-1]]
    if icu_id_param[icu_id_param.chart_time==0].shape[0]==0:
        icu_id_param=pd.concat([icu_id_param[icu_id_param.chart_time==list(set(icu_id_param.chart_time))[0]],icu_id_param],ignore_index=True)
        icu_id_param.chart_time.iloc[0]=0;icu_id_param[labelname].iloc[0]=np.nan

    for i in range(0,int(chart_time_opt[-1])):
        if icu_id_param[icu_id_param.chart_time==i].shape[0]==0:
            icu_id_param=pd.concat([icu_id_param[icu_id_param.chart_time<i],icu_id_param[icu_id_param.chart_time<i].iloc[-1].to_frame().T,icu_id_param[icu_id_param.chart_time>i]],ignore_index=True)
            icu_id_param.chart_time.iloc[i]=i;icu_id_param[labelname].iloc[i]=np.nan
    return icu_id_param
labelname = ['vent_statu']
def final_data(i):
    icu_id_param=comtest[comtest.icustay_id==i] 
    icu_id_param.sort_values(by="chart_time", inplace=True, ascending=True) 
    chart_time_opt=np.array(icu_id_param.chart_time[icu_id_param.vent_statu.notnull()]) 
    chart_time_opt=chart_time_opt[chart_time_opt>=learn_len+gap]   
    temp_return=np.ones([1,len(new_name)]); 
    if len(chart_time_opt)!=0: 
        icu_id_param=inter_learn_len(chart_time_opt,icu_id_param,labelname)  
        for jj in chart_time_opt:  
            for fore_len_l in range(0,fore_len):
                if jj-gap-learn_len-fore_len_l>=0: 
                    if (np.array((icu_id_param.loc[jj-gap-learn_len:jj-1,labelname].sum()>0)+0)[0]==1):
                        continue
                    index_1=np.ones([icu_id_param.shape[0],1])*0;
                    index_1[icu_id_param.chart_time<jj-gap-fore_len_l]=1   
                    index_2=np.ones([icu_id_param.shape[0],1])*0;
                    index_2[icu_id_param.chart_time>=jj-gap-learn_len-fore_len_l]=1 
                    temporary=icu_id_param[index_1+index_2>=2];
                    temp=np.append(np.array(temporary[['icustay_id','hadm_id','gender_0','gender_1','age','bmi']].iloc[0].to_frame().T),
                                   np.array(temporary.drop(columns=['icustay_id','hadm_id','gender_0','gender_1','age','bmi','chart_time','vent_statu','death_label','to_discharge'])).T.reshape([1,-1]))
                    temp=np.append(temp,np.array(icu_id_param[icu_id_param.chart_time==jj].vent_statu))
                    temp=np.append(temp,np.array(icu_id_param[icu_id_param.chart_time==jj].death_label))
                    temp=np.append(temp,np.array(icu_id_param[icu_id_param.chart_time==jj].to_discharge)).reshape([1,-1]) 
                    temp_return=np.append(temp_return,temp,axis=0)
    temp_return=np.delete(temp_return,0,axis=0)  
    temp_return=pd.DataFrame(temp_return)
    return temp_return

import pandas as pd
pd.set_option('mode.chained_assignment', None)
icustay_id=list(set(comtest.icustay_id))

from tqdm import tqdm  #!!!!!!!!!!
# #数据并行
import concurrent
with concurrent.futures.ProcessPoolExecutor() as executor:  
    pro=list(tqdm(executor.map(final_data,set(comtest.icustay_id)) ,total=len(set(comtest.icustay_id)),desc='progress:'))

for i in range(0,len(pro)):
    if i==0:
        input_mulit=('pro[{}]'.format(i))
    else:
        input_mulit=(input_mulit+',pro[{}]'.format(i))

p=pd.concat(eval(input_mulit),axis=0,ignore_index=True)
p.columns=new_name  
p.to_csv('p_1_4_4.csv',index=False) 
#%%
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import model_selection

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn import ensemble
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix #混淆矩阵
from numpy import *
def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)
def evaluating_indicator(y_true, y_test, y_test_value): 
    c_m = confusion_matrix(y_true, y_test)
    TP=c_m[0,0]
    FN=c_m[0,1]
    FP=c_m[1,0]
    TN=c_m[1,1]
    
    TPR=TP/ (TP+ FN) #敏感性
    TNR= TN / (FP + TN) #特异性
    BER=1/2*((FP / (FP + TN) )+FN/(FN+TP))
    
    ACC = accuracy_score(y_true, y_test)
    MCC = matthews_corrcoef(y_true, y_test)
    F1score =  f1_score(y_true, y_test)
    AUC = roc_auc_score(y_true,y_test_value[:,1])
    KAPPA=kappa(c_m)
    
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC,'KAPPA':KAPPA}
    return c 
def blo(pro_comm_Pre,jj):     
    blo_Pre=zeros(len(pro_comm_Pre)) 
    blo_Pre[(pro_comm_Pre[:,1]>(jj*0.01))]=1
    return blo_Pre

def spec_for_ser(df,icustay_id): 
    str_df=str(df)
    for i in icustay_id:
        if i==icustay_id[0]: 
            input_mulit=(str_df+"["+str_df+"['icustay_id']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['icustay_id']=={}]".format(i))
    return (pd.concat(eval(input_mulit),axis=0,ignore_index=True))
comtest = pd.read_csv("p_1_4_4.csv")
comtest.drop(['death_label','to_discharge','hadm_id'],axis=1,inplace=True)  ## 全部参数
# comtest.drop(['pao2fio2ratio_1','albumin_lab_1','bands_lab_1','bilirubin_lab_1','creatinine_lab_1','platelet_lab_1','ptt_lab_1','inr_lab_1','pt_lab_1','bun_lab_1', 'wbc_lab_1','spo2_1','so2_1','baseexcess_1','bicarbonate_1','totalco2_1','carboxyhemoglobin_1','chloride_1','calcium_1','hematocrit_1','hemoglobin_1','lactate_1','methemoglobin_1','pco2_1','ph_1','po2_1','requiredo2_1','sodium_1','potassium_1'],axis=1,inplace=True)
comtest=comtest[comtest.vent_statu<=1]
icustay_id=list(set(comtest['icustay_id']))
scaler = StandardScaler()  
comtest.iloc[:,1:comtest.shape[1]-1]=scaler.fit_transform(comtest.iloc[:,1:comtest.shape[1]-1])  
#根据患者ID划分训练集测试集
x_train_for_vail, x_test, y_train_for_vail, y_true = model_selection.train_test_split(np.array(icustay_id),range(len(icustay_id)), test_size = 0.2,random_state=10)    
x_train_for_vail=spec_for_ser('comtest',x_train_for_vail);
y_train_for_vail=x_train_for_vail.iloc[:,-1];
x_train_for_vail=x_train_for_vail.iloc[:,1:x_train_for_vail.shape[1]-1]
x_test=spec_for_ser('comtest',x_test);
y_true=x_test.iloc[:,-1];
x_test=x_test.iloc[:,1:x_test.shape[1]-1]
comm = lgb.LGBMClassifier()
comm = RandomForestClassifier()
comm = MLPClassifier()
comm = LogisticRegression()
comm = svm.SVC(probability=True)
comm = ensemble.AdaBoostClassifier()
comm = XGBClassifier()
comm = GaussianNB()
comm = BernoulliNB()
comm = MultinomialNB()
comm = KNeighborsClassifier(n_neighbors=6)
comm.fit(x_train_for_vail ,y_train_for_vail)
#%%KNN的预测
x = np.array(x_test)
pro_comm_Pre = comm.predict_proba(x)
pro_comm_Pre=comm.predict(x)
RightIndex=[]
for jj in range(100): 
    blo_comm_Pre = blo(pro_comm_Pre,jj)
    eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
    RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
RightIndex=np.array(RightIndex,dtype=np.float16)
position=np.argmin(RightIndex)  
position=position.mean()
blo_comm_Pre = blo(pro_comm_Pre,position)  
eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
print(eva_comm)  
RightIndex=[]
for jj in range(100): 
    blo_comm_Pre = blo(pro_comm_Pre,jj)
    eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
    RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
RightIndex=np.array(RightIndex,dtype=np.float16)
position=np.argmin(RightIndex)  
position=position.mean()

blo_comm_Pre = blo(pro_comm_Pre,position) 

eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
print(eva_comm)  
#计算模型性能95%置信区间  
y_true=np.array(y_true)
n_bootstraps = 5000

bootstrapped_scores=[]
EVA_ACC_CI=[];EVA_AUC_CI=[]
EVA_MCC_CI=[];EVA_F1_score_CI=[]

EVA_BER_CI=[];EVA_KAPPA_CI=[]
EVA_TPR_CI=[];EVA_TNR_CI=[]

for i in range(n_bootstraps):
    indices=np.random.randint(0,len(pro_comm_Pre[:,1]) - 1,len(pro_comm_Pre[:,1]))
    if len(np.unique(y_true[indices])) < 2:
        continue
    eva_CI = evaluating_indicator(y_true=y_true[indices], y_test=blo_comm_Pre[indices], y_test_value=pro_comm_Pre[indices])
    EVA_AUC_CI.append(eva_CI['AUC']);
    EVA_ACC_CI.append(eva_CI['ACC']);
    EVA_MCC_CI.append(eva_CI['MCC']);
    EVA_F1_score_CI.append(eva_CI['F1_score']);EVA_BER_CI.append(eva_CI['BER']);EVA_KAPPA_CI.append(eva_CI['KAPPA']);
    EVA_TPR_CI.append(eva_CI['TPR']);EVA_TNR_CI.append(eva_CI['TNR']);
sorted_AUC_scores = np.array(EVA_AUC_CI); sorted_AUC_scores.sort()          
sorted_ACC_scores = np.array(EVA_ACC_CI); sorted_ACC_scores.sort()
sorted_MCC_scores = np.array(EVA_MCC_CI); sorted_MCC_scores.sort()
sorted_F1_score_scores = np.array(EVA_F1_score_CI); sorted_F1_score_scores.sort()
sorted_BER_scores = np.array(EVA_BER_CI); sorted_BER_scores.sort()
sorted_KAPPA_scores = np.array(EVA_KAPPA_CI); sorted_KAPPA_scores.sort()
sorted_TPR_scores = np.array(EVA_TPR_CI); sorted_TPR_scores.sort()
sorted_TNR_scores = np.array(EVA_TNR_CI); sorted_TNR_scores.sort()
print("Confidence interval for the AUC: [{:0.6f} - {:0.6}]".format(sorted_AUC_scores[int(0.025 * len(sorted_AUC_scores))], sorted_AUC_scores[int(0.975 * len(sorted_AUC_scores))]))
print("Confidence interval for the ACC: [{:0.6f} - {:0.6}]".format(sorted_ACC_scores[int(0.025 * len(sorted_ACC_scores))], sorted_ACC_scores[int(0.975 * len(sorted_ACC_scores))]))
print("Confidence interval for the BER: [{:0.6f} - {:0.6}]".format(sorted_BER_scores[int(0.025 * len(sorted_BER_scores))], sorted_BER_scores[int(0.975 * len(sorted_BER_scores))]))
print("Confidence interval for the F1_score: [{:0.6f} - {:0.6}]".format(sorted_F1_score_scores[int(0.025 * len(sorted_F1_score_scores))], sorted_F1_score_scores[int(0.975 * len(sorted_F1_score_scores))]))
print("Confidence interval for the KAPPA: [{:0.6f} - {:0.6}]".format(sorted_KAPPA_scores[int(0.025 * len(sorted_KAPPA_scores))], sorted_KAPPA_scores[int(0.975 * len(sorted_KAPPA_scores))]))
print("Confidence interval for the MCC: [{:0.6f} - {:0.6}]".format(sorted_MCC_scores[int(0.025 * len(sorted_MCC_scores))], sorted_MCC_scores[int(0.975 * len(sorted_MCC_scores))]))
print("Confidence interval for the TNR: [{:0.6f} - {:0.6}]".format(sorted_TNR_scores[int(0.025 * len(sorted_TNR_scores))], sorted_TNR_scores[int(0.975 * len(sorted_TNR_scores))]))
print("Confidence interval for the TPR: [{:0.6f} - {:0.6}]".format(sorted_TPR_scores[int(0.025 * len(sorted_TPR_scores))], sorted_TPR_scores[int(0.975 * len(sorted_TPR_scores))]))




