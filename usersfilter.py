
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
def eachavguserrlist(avglist,allincsvfile,newincsvfile):
    df_alllist = pd.read_csv(allincsvfile, header=0, parse_dates=['date'])
    df_newlist = pd.read_csv(newincsvfile, header=0, parse_dates=['date'])

    allrecords = df_alllist.sort_values(by=["insight_id", "date","count"])
    newrecords = df_newlist.sort_values(by=["insight_id", "date","count"])
    idlist=allrecords["insight_id"].drop_duplicates()
    shiftdaylist={}
    bestscoreindexlist={}
    for insight_id in idlist:
        print(insight_id)
        allpickuprec = allrecords[(allrecords['insight_id'] == insight_id)]
        newpickuprec = newrecords[(newrecords['insight_id'] == insight_id)]
        fillterrec=pd.merge(allpickuprec, newpickuprec, how="left", on="date")
        if len(fillterrec)<5:
            shiftday=0
            bestscoreindex=0
            shiftdaylist[insight_id] = shiftday
            bestscoreindexlist[insight_id] = bestscoreindex
            continue
        avgcount=(fillterrec["count_y"]/fillterrec["count_x"])
        rscorelist=[]
        for shiftsize in range(1,20):
           rscore1,rscore2=randomForest_process(avgcount,shiftsize)
           rscorelist.append(rscore1)
        bestrscore=min(rscorelist)
        shiftday=get_shiftday(avglist[insight_id],avgcount)
        bestscoreindex=rscorelist.index(bestrscore)
        shiftdaylist[insight_id]=shiftday
        bestscoreindexlist[insight_id]=bestscoreindex
    return shiftdaylist,bestscoreindexlist

def getConfidenceinterval(valafterlist):
        alpha = 0.95
        valafterlist=valafterlist.fillna(0)
        mean_val = np.mean(valafterlist.values)
        sem_val = stats.sem(valafterlist.values)
        if (sem_val != sem_val):
            sem_val=0
        c_min, c_max = stats.t.interval(alpha, len(valafterlist.values) - 1, loc=mean_val, scale=sem_val)
        return c_max, c_min


def get_shiftday(avgscore,avgcount):
    max,min=getConfidenceinterval(avgcount)
    offset=0

    for count in avgcount:
      if(count != count):
        shiftday=offset
        offset+=1
        continue
      if min> count:
          shiftday=offset
          break
      if avgscore> count:
        shiftday=offset
        break
      offset+=1
    return shiftday


def randomForest_process(avglist,shiftsize):

    modlen=len(avglist)%2
    avglen=int(len(avglist)/2)
    if modlen>0:
         avglen-=1
    y_train=avglist[0:avglen].shift(-shiftsize)
    y_test=avglist[0:avglen]
    X_train=avglist[avglen:avglen+avglen]
    X_test = avglist[avglen:avglen+avglen]

    X_train=X_train.fillna(0)
    y_train=y_train.fillna(0)
    X_test=X_test.fillna(0)
    y_test=y_test.fillna(0)
    X_train = X_train[:, None]
    X_test = X_test[:, None]
    y_test = y_test[:, None]

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.grid_search import GridSearchCV

    parameters = {
        'n_estimators': [5, 10, 20, 30, 50,100],
        'n_jobs': [-1],
    }
    forest = RandomForestRegressor()
    cv = GridSearchCV(forest, parameters)
    cv.fit(X_train, y_train)
    y_train_pred = cv.predict(X_train)
    y_test_pred = cv.predict(X_test)

#    return (y_train-y_train_pred).mean(), (y_test-y_test_pred).mean()
    return (y_train_pred-y_train).mean(), (y_test_pred-y_test).mean()

def write_scoreprocess(shifydaylist,bestscorelist,outcsvfile):
    df=pd.Series(shifydaylist)
    df.to_csv(outcsvfile+"/appdshiftday_totalavgscore.csv")
    df = pd.Series(bestscorelist)
    df.to_csv(outcsvfile+ "/appshiftday_randomForest.csv")


def filter_process(appavglist,allincsvfile,newincsvfile,outcsvfile):
    shiftdaylist,bestscorelist=eachavguserrlist(appavglist,allincsvfile,newincsvfile)
    write_scoreprocess(shiftdaylist,bestscorelist,outcsvfile)
def learn_allappavgprocess(allcsvfile,newcsvfile):
    appidavglist = {}
    df_alllist = pd.read_csv(allcsvfile, header=0, parse_dates=['date'])
    df_newlist = pd.read_csv(newcsvfile, header=0, parse_dates=['date'])
    allrecords = df_alllist.sort_values(by=["insight_id", "date","count"])
    newrecords = df_newlist.sort_values(by=["insight_id", "date","count"])
    idlist=allrecords["insight_id"].drop_duplicates()

    for insight_id in idlist:
        allpickuprec = allrecords[(allrecords['insight_id'] == insight_id)]
        newpickuprec = newrecords[(newrecords['insight_id'] == insight_id)]
        avgcount = (newpickuprec["count"].sum()/allpickuprec["count"].sum())/len(allpickuprec)

        fillterrec=pd.merge(allpickuprec, newpickuprec, how="left", on="date")
        avgcount2=((fillterrec ["count_y"]/fillterrec["count_x"]).sum())/len(fillterrec)
        appidavglist[insight_id]=avgcount2
    return  appidavglist

def main(argv):

    argvs = sys.argv
    argc = len(argvs)

    if len(argv) != 3:
        sys.exit("Usage: %s allinputcsvpath  newinputcsvpath outputcsvpath" % sys.argv[0])
        quit()
    allincsvfile = argv[0]
    newincsvfile=argv[1]
    outcsvfile = argv[2]
    appavglist=learn_allappavgprocess(allincsvfile,newincsvfile)
    filter_process(appavglist,allincsvfile,newincsvfile,outcsvfile)


if __name__ == "__main__":
    main(sys.argv[1:])
