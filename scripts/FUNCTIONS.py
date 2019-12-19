#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import division

__author__ = 'Aanchal'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors
import random
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, MeanShift, estimate_bandwidth
import seaborn as sns

from os.path import isfile, join, exists
from os import listdir, mkdir

import scipy.cluster.hierarchy as shc
from shutil import copyfile
from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.metrics import pairwise_distances

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm
import random
import seaborn as sns; sns.set()
from pylab import hist, exp 
from scipy.optimize import curve_fit
import pickle


        

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    '''print and plot the confusion matrix.
    Normalization can be applied by setting `normalize=True`'''
   
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #print(unique_labels(y_true, y_pred))
    classes = ['Resistant','Sensitive','Hump','Rising']#classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='CCLE',
           xlabel='GDSC')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def dic2csv(dic, filename):
    '''write a dictionary to csv file'''
    pd.DataFrame.from_dict(data=dic, orient='index').to_csv(filename, header=False)
    
def getSensitivityMetrics(FeatureMatrix_sp_ic):
    ''' get  ic50, slope and cell death at GDSC mid dosage for CCLE and GDSC datasets'''
    [ic50_ccle, m_ccle]=get_ic50slope('CCLE')
    [ic50_gdsc,m_gdsc]=get_ic50slope('GDSC')

    ccle_ic50_list=[]
    gdsc_ic50_list=[]
    ccle_m_list=[]
    gdsc_m_list=[]
    ccle_cdmid_list=[]
    gdsc_cdmid_list=[]
    ccle_cdmax_list=[]
    gdsc_cdmax_list=[]
    for i, row in FeatureMatrix_sp_ic.iterrows():

        ind_cl=np.where( ic50_ccle.index == row['Standard cell line name']  )[0]
        ind_dr=np.where( ic50_ccle.columns == row['Standard drug name'] )[0]
        #print(row['Standard cell line name'] ,  row['Standard drug name'] , ind_cl , ind_dr)
        ccle_ic50_list.append (  ic50_ccle.values[ind_cl , ind_dr  ][0] )
        ccle_m_list.append (  m_ccle.values[ind_cl , ind_dr  ][0] )
        ccle_cdmid_list.append (  getResponse(np.log2(row['mid_dosage']),ic50_ccle.values[ind_cl , ind_dr  ][0],m_ccle.values[ind_cl , ind_dr  ][0]) )
        ccle_cdmax_list.append (  getResponse(np.log2(row['max_dosage']),ic50_ccle.values[ind_cl , ind_dr  ][0],m_ccle.values[ind_cl , ind_dr  ][0]) )


        ind_cl=np.where( ic50_gdsc.index== row['Standard cell line name'])[0]
        ind_dr=np.where( ic50_gdsc.columns ==row['Standard drug name'] )[0]#str(gdse_dr_id[0]) )[0]
        gdsc_ic50_list.append (   ic50_gdsc.values[ind_cl , ind_dr  ][0] )
        gdsc_m_list.append (   m_gdsc.values[ind_cl , ind_dr  ][0] )
        gdsc_cdmid_list.append (   getResponse(np.log2(row['mid_dosage']), ic50_gdsc.values[ind_cl , ind_dr  ][0], m_gdsc.values[ind_cl , ind_dr  ][0] )  )
        gdsc_cdmax_list.append (   getResponse(np.log2(row['max_dosage']), ic50_gdsc.values[ind_cl , ind_dr  ][0], m_gdsc.values[ind_cl , ind_dr  ][0] )  )




    FeatureMatrix_sp_ic['ic50_ccle']=ccle_ic50_list
    FeatureMatrix_sp_ic['ic50_gdsc']=gdsc_ic50_list
    FeatureMatrix_sp_ic['m_ccle']=ccle_m_list
    FeatureMatrix_sp_ic['m_gdsc']=gdsc_m_list
    FeatureMatrix_sp_ic['cd@midDosageOfGdsc_ccle']=ccle_cdmid_list
    FeatureMatrix_sp_ic['cd@midDosageOfGdsc_gdsc']=gdsc_cdmid_list
    FeatureMatrix_sp_ic['cd@maxDosageOfGdsc_ccle']=ccle_cdmax_list
    FeatureMatrix_sp_ic['cd@maxDosageOfGdsc_gdsc']=gdsc_cdmax_list
    
    return FeatureMatrix_sp_ic

def get_ic50s_asFeatures(FeatureMatrix):
    ''' get  ic50, slope and cell death at GDSC mid dosage for CCLE and GDSC datasets'''
    [ic50_ccle, m_ccle]=get_ic50slope('CCLE')
    [ic50_gdsc,m_gdsc]=get_ic50slope('GDSC')

    ccle_ic50_list=[]
    gdsc_ic50_list=[]
    
    for i, row in FeatureMatrix.iterrows():

        ind_cl=np.where( ic50_ccle.index == row['Standard cell line name']  )[0]
        ind_dr=np.where( ic50_ccle.columns == row['Standard drug name'] )[0]
        #print(row['Standard cell line name'] ,  row['Standard drug name'] , ind_cl , ind_dr)
        ccle_ic50_list.append (  ic50_ccle.values[ind_cl , ind_dr  ][0] )

        ind_cl=np.where( ic50_gdsc.index== row['Standard cell line name'])[0]
        ind_dr=np.where( ic50_gdsc.columns ==row['Standard drug name'] )[0]#str(gdse_dr_id[0]) )[0]
        gdsc_ic50_list.append (   ic50_gdsc.values[ind_cl , ind_dr  ][0] )



    FeatureMatrix['ic50_ccle']=ccle_ic50_list
    FeatureMatrix['ic50_gdsc']=gdsc_ic50_list
    
    return FeatureMatrix

def get_ic50slope(dataname):
    '''Get IC50 and slope matrices for datasets (CCLE or GDSC)'''
    [dic_cl,dic_dr]=getDictionary2(dataname)
    
    if (dataname=='CCLE'):
        ic50=pd.read_csv('../data/drug_response/CCLE/ccle_all_abs_ic50_bayesian_sigmoid.csv', index_col=0) 
        m=pd.read_csv('../data/drug_response/CCLE/ccle_slope_bayesian_sigmoid.csv', index_col=0) 

    elif(dataname=='GDSC'):
        ic50=pd.read_csv('../data/drug_response/GDSC/gdsc_all_abs_ic50_bayesian_sigmoid.csv', index_col=0) 
        m=pd.read_csv('../data/drug_response/GDSC/gdsc_slope_bayesian_sigmoid.csv', index_col=0) 
        
        ic50.index = ic50.index.map(str)
        ic50.columns = ic50.columns.map(int)
        m.index = m.index.map(str)
        m.columns = m.columns.map(int)
    else:
        print("The function has not currently been programmed to return IC50 and slope values for datasets other than CCLE and GDSC!")
        exit()
        
    ic50=ic50.rename(index=dic_cl)
    ic50=ic50.rename(columns=dic_dr)
    m=m.rename(index=dic_cl)
    m=m.rename(columns=dic_dr)
    
    return ic50, m
    

def get_common_drugs_metadata(drug_metadata_filename,sheet_name):
    '''Get min dosage, max dosage, overlapping range, GDSc dosage length for drugs common in CCLE and GDSC datasets'''
    
    dr_metadata_file=pd.read_excel(drug_metadata_filename, sheetname=sheet_name)
    dr_metadata=dr_metadata_file

    dr_metadata['GDSC min dosage']=np.log2(dr_metadata['GDSC min dosage'])
    dr_metadata['GDSC max dosage']=np.log2(dr_metadata['GDSC max dosage'])
    dr_metadata['CCLE min dosage']=np.log2(dr_metadata['CCLE min dosage'])
    dr_metadata['CCLE max dosage']=np.log2(dr_metadata['CCLE max dosage'])
    
    
    dr_metadata['name']=dr_metadata[['name']]
    dr_metadata['max_start']=dr_metadata[["GDSC min dosage", "CCLE min dosage"]].max(axis=1)
    dr_metadata['min_end']=dr_metadata[["GDSC max dosage", "CCLE max dosage"]].min(axis=1)
    dr_metadata['dosage range overlap']=dr_metadata['min_end']-dr_metadata['max_start']
    dr_metadata=dr_metadata.sort_values(by='name')
    dr_metadata=dr_metadata.set_index(dr_metadata['name'])
    dr_metadata.rename(columns={'name': 'Standard drug name'}, inplace=True)

    dr_metadata['GDSC dosage length']=dr_metadata['GDSC max dosage']-dr_metadata['GDSC min dosage']
    
    del dr_metadata['max_start']
    del dr_metadata['min_end']
    
    return dr_metadata


def split_bimodal(xh,yh,mus):
    '''Get the inflexion/saddle/mid point between two gaussians, given the distribution and their means'''
    
    xh_sliced=  xh[(xh>=mus[0]) & (xh<=mus[1]) ]

    yh_sliced=yh[(xh>=mus[0]) & (xh<=mus[1])]
    mid_pt=xh_sliced[np.where(yh_sliced==np.min(yh_sliced))] # or mid=np.median(xh_sliced ) #https://www.researchgate.net/post/How_to_separate_a_bimodal_distribution_of_response_times_RT
    #plt.axvline(x=mid_pt ,color='r', linestyle='--', label='threshold')
    
    return mid_pt


def gauss(x,mu,sigma,A):
    '''get gaussian profile given mean, std and amplitude'''
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    '''get sum/mixture of 2 gaussians'''
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)


# In[ ]:
def fit_mixtureOfGaussians(data, guess=[]):
    '''fit mixture of gaussians on bimodal data with initial parameter estimates "guess" '''
    data=data.dropna().values.flatten()
    yh,xh,_=hist(data,50,alpha=.3,label='data')
    
    xh=(xh[1:]+xh[:-1])/2 # for len(x)==len(y)

    #expected=(0.01,.2,25,0.1,.2,12)
    if(guess):
        params,cov=curve_fit(bimodal,xh,yh, maxfev=50000,p0=guess) #params optimal value of parameters of function passed ("biomodel" here)
    else:
        params,cov=curve_fit(bimodal,xh,yh)
    #sigma=sqrt(diag(cov))#variance f estimate parameters
    
    plt.plot(xh,bimodal(xh,*params),color='red',lw=3,label='model')
    plt.legend()
    print("mu, sigma, Amp for 2 modes=> "+str(params) ) 

    mus=[params[0], params[3]]
    stds=[params[1], params[4]]
        
    return mus,stds, params,xh,yh
    
    


def plotVectorsGetCorr_seaborn( col1, col2, df ,corrType,  col3=[], fs=10, alpha=1):
    '''plot two vectors to observe if they correlate or not (using seaborn library) '''
    
    correl=df[col1].corr(df[col2],method=corrType)
    
    
    if (col3):
        ax=sns.scatterplot(x=col1, y=col2, data=df, hue=col3,  marker='o', alpha=alpha, legend=False)#size=col3, legend="brief",
        print("hiii")
        #plt.scatter(df1[col1] , df2[col2], s=100*df3[col3], c=100*df3[col3])
    else:
        ax=sns.scatterplot(x=col1, y=col2, data=df, marker='o', alpha=alpha)
        
    ax.set_title(corrType+" corr="+ str(correl.round(2)  ))
    plt.xlabel(col1, fontsize=fs)
    plt.ylabel(col2, fontsize=fs)
    plt.title( corrType+"corr="+ str(correl.round(2)), fontsize=fs )
    return correl.round(2)



def plotVectorsGetCorr(df1, col1, df2, col2, annotationCol,corrType, df3=[], col3=[]):
    '''plot two vectors to observe if they correlate or not  '''
    #find corr
    correl=df1[col1].corr(df2[col2],method=corrType)
    
    #plot vectors
    
    if (col3):
        plt.scatter(df1[col1] , df2[col2], s=100*df3[col3], c=100*df3[col3])
    else:
        plt.scatter(df1[col1] , df2[col2]) 
        
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title( corrType+"corr="+ str(correl.round(2)) )
                
    
    for i, txt in enumerate(annotationCol):
        plt.text(df1.loc[txt,col1], df2.loc[txt,col2], txt, color='grey')
    
    return correl


def get5dosagePoints(log2_dosage):
    '''get 5 equally distributed dosage values given a drug dosage range (here on log scale)'''
    mindosage=np.nanmin(log2_dosage)
    maxdosage=np.nanmax(log2_dosage)
    middosage=mindosage + (maxdosage-mindosage)/2
    ab_dosage=mindosage+(middosage-mindosage)/2
    bc_dosage=middosage+(maxdosage-middosage)/2
    
    return mindosage, ab_dosage, middosage, bc_dosage, maxdosage

def get5DiscretePointsAtFittedCurves(mindosage, ab_dosage, middosage, bc_dosage, maxdosage,beta0,beta1):
    '''get start and end points of HDI at 5 discrete points '''
    
    response_func=np.vectorize(getResponse)  
    
    a_st=getHDI( response_func(mindosage,beta0,beta1), 0.05 )[1][0]
    a_end=getHDI( response_func(mindosage,beta0,beta1), 0.05 )[2][0]
    ab_st=getHDI( response_func(ab_dosage,beta0,beta1), 0.05 )[1][0]
    ab_end=getHDI( response_func(ab_dosage,beta0,beta1), 0.05 )[2][0]
    b_st=getHDI( response_func(middosage,beta0,beta1), 0.05 )[1][0]
    b_end= getHDI( response_func(middosage,beta0,beta1), 0.05 )[2][0] 
    bc_st=getHDI( response_func(bc_dosage,beta0,beta1), 0.05 )[1][0]
    bc_end=getHDI( response_func(bc_dosage,beta0,beta1), 0.05 )[2][0]
    c_st=getHDI( response_func(maxdosage,beta0,beta1), 0.05 )[1][0]
    c_end= getHDI( response_func(maxdosage,beta0,beta1), 0.05 )[2][0] 
    
    curve1=np.array( [a_st,ab_st,b_st,bc_st,c_st] )
    curve2=np.array( [a_end,ab_end,b_end,bc_end,c_end] )
    
    return curve1, curve2
        
def copyPastePlots(src_folder, dest_folder, algo, superalgofoldername,level, k, Intersection_names):
    '''copy cl-dr plots from source to destination folder (used in kms clustering)'''
    
    if not exists(dest_folder+'/'+superalgofoldername+algo+'_k='+str(k)+'/'):
            mkdir(dest_folder+'/'+superalgofoldername+algo+'_k='+str(k)+'/')
    for cluster_no in np.unique(Intersection_names[[algo+'_'+level+'label']]):#in range(k):
        if not exists(dest_folder+'/'+superalgofoldername+algo+'_k='+str(k)+'/'+'/cluster_'+str( cluster_no) ):
            mkdir(dest_folder+'/'+superalgofoldername+algo+'_k='+str(k)+'/'+'/cluster_'+str( cluster_no) )
    
    ################### COPY FILES from source folder to a folder corresponding to each cluster
    for ind, row in Intersection_names.iterrows():
        filename=row['Standard cell line name']+','+row['Standard drug name']+'.png'
        copyfile(src_folder+filename, dest_folder+'/'+superalgofoldername+algo+'_k='+str(k)+'/'+'/cluster_'+str(row[algo+'_'+level+'label'])+'/'+filename)
    print("curves saved in folder: ",dest_folder+'/'+superalgofoldername+algo+'_k='+str(k))

    
def copyPastePlotsSubclustering(src_folder, dest_folder, algo, k,cluster_labels, Intersection_names):
    '''get start and end points of HDI at 5 discrete points (used in level 2 clustering) '''
    
    if not exists(dest_folder+algo+'_k=7'+'/subcluster_/'):
            mkdir(dest_folder+algo+'_k=7'+'/subcluster_')
    for cluster_no in np.unique(cluster_labels):#in range(k):
        if not exists(dest_folder+algo+'_k=7/subcluster_/'+'/cluster_'+str( cluster_no) ):
            mkdir(dest_folder+algo+'_k=7/subcluster_'+'/cluster_'+str( cluster_no) )



    ###################                 COPY FILES from source folder to a fodler corresponding to each cluster
    for ind, row in  Intersection_names.iterrows():
        filename=row['Standard cell line name']+','+row['Standard drug name']+'.png'
        copyfile(src_folder+filename, dest_folder+algo+'_k=7/subcluster_'+'/cluster_'+str(row[algo+'_Sublabel']) +'/'+filename)
# In[ ]:


import similaritymeasures
def getCurveSimilarities(x_ccle,y_ccle,x_gdsc,y_gdsc):
    '''get similarity (area and dtw) between dose-response viability (curves) values  '''
    
    exp_data = np.zeros((len(x_ccle), 2))
    num_data = np.zeros((len(x_gdsc), 2))
    
    exp_data[:, 0] = x_ccle
    exp_data[:, 1] = y_ccle
    num_data[:, 0] = x_gdsc
    num_data[:, 1] = y_gdsc
    
    # quantify the difference between the two curves using area between two curves
    # : using older version of similaritymeasures in the form of code (last of this file) in FUNCTIONS file since newer version gives wrong area
    area =area_between_two_curves(exp_data, num_data)  #similaritymeasures.area_between_two_curves(exp_data, num_data)
    # quantify the difference between the two curves using Dynamic Time Warping distance
    dtw, d = similaritymeasures.dtw(exp_data, num_data)
    
    return (area,dtw)


# In[2]:


def clusteringBoxplot(combined, colname, features, letter,ypos,violin, orderby=None):
    '''plot boxplot (violin=0) or violon plot (violon=1): "colname" on xaxis and "features" on y axis in "combined" dataframe with xlabels having prefix "letter" and number of data points shown in 1 boxplot at "ypos" '''

    ###################                 GROUPED BOXPLOT
    plt.figure()
    noe_percluster=combined[colname].value_counts().reset_index().sort_values(by='index').values[:,1]
    dd=pd.melt(combined,id_vars=[colname],value_vars=features,var_name='features')


    if(violin==1):
        bp=sns.violinplot(x=colname,y='value',data=dd,hue='features',  palette="colorblind",fliersize=0, order=orderby )
    else:
        bp=sns.boxplot(x=colname,y='value',data=dd,hue='features',  palette="colorblind",fliersize=0, order=orderby )
        #bp.set(xlabel=colname,ylabel=ylabel)
    bp=sns.stripplot(y='value', x=colname,data=dd,jitter=True,dodge=True,marker='.',alpha=0.5,hue='features',color='grey')

    # setting xlabels
    ''''xlabels=[]
    print(bp.get_xticklabels())
    for i in (bp.get_xticklabels() ):
        print(str(i))
        if("." in str(i).split(',')[2]):
            suffix=letter+ str(i).split(',')[2].split('.')[0][2:]
        else:
            suffix=letter+str(i).split(',')[2][2]
        #print(letter+ str(i).split(',')[2].split('.')[0][2:]  )
        #print(letter+str(i).split(',')[2][2] )
        xlabels.append(suffix )
    bp.set_xticklabels(xlabels)'''

    # get legend information from the plot object
    handles, labels = bp.get_legend_handles_labels()
    # specify just one legend
    l = plt.legend(handles[0:5], labels[0:5])
    lgd=plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

    nobs = noe_percluster
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n:" + i for i in nobs]

    # Add it to the plot
    pos = range(len(nobs))
    for tick,label in zip(pos,bp.get_xticklabels()):
        bp.text(pos[tick]-0.3, ypos, nobs[tick],
        verticalalignment='bottom', size='small', color='b', weight='semibold')


    #plt.savefig(dest_folder+algo+'_k='+str(k)+'_nof='+str(len(features)),  dpi=300 ,bbox_extra_artists=(lgd,), bbox_inches='tight')

        
    
def cluster(method, FM,k):
    '''perform clustering of samples in "FM" (initialize k for kmeans, else k=0) '''
    
    '''if (method=='PCAnkmeans'):
        k=f.silhouette_based_cluster_selection(PCA(n_components=2).fit_transform(FM),  [2, 3, 4, 5, 6, 7 , 8] ,method)
    else:
        
        k=f.silhouette_based_cluster_selection(FM.values,  [2, 3, 4, 5, 6, 7 , 8] ,method)'''
    
       
        
    if (method=='kmeans'):
        if(k==0):
            k=silhouette_based_cluster_selection(FM.values,  [2, 3, 4, 5, 6, 7 , 8] ,method)
        
        model = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(FM)
    elif (method=='PCAnkmeans'):
        k=silhouette_based_cluster_selection(PCA(n_components=2).fit_transform(FM),  [2, 3, 4, 5, 6, 7 , 8] ,method)
        reduced_data = PCA(n_components=2).fit_transform(FM)
        model= KMeans(init='k-means++', n_clusters=k, n_init=10).fit(reduced_data)
    elif(method=='agglo'):
        #k=f.silhouette_based_cluster_selection(FM.values,  [2, 3, 4, 5, 6, 7 , 8] ,method)
        model = AgglomerativeClustering(n_clusters=8,linkage='average', affinity='euclidean').fit(FM)  #affinity=pearson_affinity).fit((FM) ) #'can use affinity='correaltion' also
        dendrogram = sch.dendrogram(sch.linkage(FM,method='ward'))
        k=len( np.unique( model.labels_ ))
        #k=model.n_clusters_
        #n_clusters_ = len(model.cluster_centers_indices_)
    elif(method=='AffinityPropagation'):
        #k=f.silhouette_based_cluster_selection(FM.values,  [2, 3, 4, 5, 6, 7 , 8] ,method)
        model = AffinityPropagation().fit(FM)
        k=len( np.unique( model.labels_ ))#k=len(np.unique(model.cluster_centers_) )
    elif(method=='MeanShift'):
        #k=f.silhouette_based_cluster_selection(FM.values,  [2, 3, 4, 5, 6, 7 , 8] ,method)
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(FM)#, quantile=0.2, n_samples=500)
        print('BW= ',bandwidth)
        #new_BW=0.225
        #print('new BW=', new_BW)
        model = MeanShift(bandwidth=bandwidth , bin_seeding=True).fit(FM)
        k=len( np.unique( model.labels_ ))#k=len((model.cluster_centers_) )
    
    #centroids = model.cluster_centers_
    return (k,model.labels_)


def silhouette_based_cluster_selection(X,range_n_clusters,method):
    '''get no of clusters (from interval "n_clusters") based on silhouette index by clustering samples in "X" using clustering method "method" '''
    
    best_si=-1
    noc=0
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        
        if(silhouette_avg>best_si):
            best_si=silhouette_avg
            noc=n_clusters
            
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()
    return noc
    
def getCommonPairs(datasets_list, rawFileDir,cl_col_name, drug_col_name, dosage_col_name,response_col_name):
    '''get common cell line-drug pairs between datasets listed in "datasets_list" by reading the standard raw files in "rawFileDir" directory '''
    cldr_names=[]
    l=[]
    for i in range( len (datasets_list) ):
        print(i)
        data_raw=pd.read_csv(rawFileDir+datasets_list[i]+"_dose_response.csv")
        #if (datasets_list[i]=='GDSC'):
            #data_raw=pd.read_csv(rawFileDir+datasets_list[i]+"_rawFile_newFormat_FINAL_no5dosageDrugs.csv")
        #print(datasets_list[i], [datasets_list[i]])
        data_raw=data_raw.assign(dataset= [datasets_list[i]]*(len(data_raw)))
        l.append(data_raw[[cl_col_name, drug_col_name, dosage_col_name, response_col_name, 'dataset']])
        cldr_names.append( data_raw[[cl_col_name, drug_col_name]] )

        if(i>0):
            Intersection_names=cldr_names[i].merge(Intersection_names)   
        else:
            Intersection_names=cldr_names[i]

    Intersection_names=pd.DataFrame(Intersection_names.drop_duplicates().values, columns=[cl_col_name, drug_col_name])  

    data_merged=pd.concat(l) # this is the full merged dataset with all pairs (may not be in common set)
    
    Intersection=data_merged.merge(Intersection_names, on=[cl_col_name, drug_col_name])
    Intersection_df=Intersection.drop_duplicates()
     
    return ( Intersection_df,Intersection_names)
    #return data_merged # to get full merged file

def getCommonPairs_CCLEnomenclature(names_ccle, names_gdsc, common_CL_info,common_DR_info):
    
    for i in range(len(common_CL_info.values[:,0])):
        gid=common_CL_info.values[i,0]
        names_gdsc['Cell_line_name']=(names_gdsc['Cell_line_name'].replace( gid  ,common_CL_info.values[i,2]) )

    for i in range(len(common_DR_info.values[:,0])):
        gid=common_DR_info.values[i,0]
        names_gdsc['Drug _Name']=(names_gdsc['Drug _Name'].replace( gid  ,common_DR_info.values[i,3]) )

    Intersection=names_gdsc.merge(names_ccle)
    return Intersection

    
def scaler(x, min, max, a, b):
    '''scale a vector to range [a-b] given the vector "x" and its min and max '''
    return ((b-a)*(x-min)/(max-min)) + a


def getResponse(x,ic50,m):
    ''' Get cell viability at dosage x (x should be in log2 scale)'''
    a1=(ic50-x) *m
    if(a1>1023):
        a=1023
    else:
        a=math.pow(2,a1)
        
    return (  1/(1+a  )    )

def get_pairInfo(Intersection, clname,drname, sample_col_name, drug_col_name, dosage_col_name, response_col_name):
    '''get log2 dosage, response and dataset source for a specific pair in "Intersection" dataframe'''
    indices_cl= Intersection.index[Intersection[ sample_col_name] == clname] 
    indices_dr= Intersection.index[Intersection[drug_col_name] == drname]
    ind=Intersection.index[indices_cl  & indices_dr]
    
    pair=pd.DataFrame(Intersection.values[ind,:], columns=[sample_col_name, drug_col_name, dosage_col_name, response_col_name,'Label'])  
    
    dosages=np.array ( pair[dosage_col_name] )
    log2_dosage= np.log2(  dosages.astype('float64') )
    responses=pair[response_col_name]
    res_0to1 = scaler(responses, 0.0, -100.0, 0.0, 1.0)
    label=pair['Label']
    
    return (log2_dosage, res_0to1, label)

def forceAspect(ax,aspect=1):
    '''set aspect ration in plot'''
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

   

def plotFittedCurves(log2_dosage,res_0to1, beta0,beta1, clname, drname, label, out_dir):
    '''plot and save a curve fitted using Baysina sigmoid curve fitting using the ic50 (beta0) and slope(beta1) given for a pair'''
    plt.figure()
    #ax = fig.add_subplot(111)

    response_func=np.vectorize(getResponse)
    
    delta=0.5
    
    np.random.seed(0)
    indices=np.random.randint(0,15000, size = 100)
    xs=np.linspace(-13,5,30)#log2_dosage# min=np.log2( Intersection['Doses (uM)'].min() )-1.7, max=np.log2( Intersection['Doses (uM)'].max() )+1.7
    
    colors=["black", "gold", "green"]
    cmap = matplotlib.colors.ListedColormap(colors)
    for i in range(len(indices)):
        plt.plot(xs, response_func(xs,beta0[indices[i]],beta1[indices[i]])  , alpha=0.1, c='grey' ,zorder=1  )########
    plt.ylim(0-delta,1+delta)
     
    
    plt.scatter(log2_dosage, res_0to1, c = label, cmap=cmap,zorder=2)
    #plt.show()
    #forceAspect(ax,aspect=1)
    
    plt.savefig(out_dir+clname+','+drname+'.png')


# In[9]:


def getDictionary(dataname, int_flag=0):
    '''Map dataset specific cell-line and drug names to Standard cell line names and Standard drug names
    int_flag: if 1, values to be maped are integers then dic wont be converted to string'''
    
    metadata_filename="../data/metadata_2019_08_15_aanchal.xlsx"
    
    if(dataname=='CCLE'):
        cl_colname_2b_mapped=dataname+' name'
        dr_colname_2b_mapped=dataname+' name'
        drug_col_name='Compound' #
        cl_col_name='CCLE Cell Line Name' #
        dosage_col_name='Doses (uM)'
        response_col_name='Activity Data (median)'
    else:
        cl_colname_2b_mapped=dataname+' id'
        dr_colname_2b_mapped=dataname+' id (1)'
        dr_colname_2b_mapped_2=dataname+' id (2)'
        drug_col_name='DRUG_ID'
        cl_col_name='COSMIC_ID'
        dosage_col_name='dose'
        response_col_name='response'
        
    cell_line_metadata =pd.read_excel(metadata_filename,sheetname="cell line")
    drug_metadata =pd.read_excel(metadata_filename,sheetname="drug")

    raw=pd.read_csv("../data/drug_response/"+dataname+"/"+dataname+"_dose_response_scores.tsv", sep="\t")
    raw=raw[[cl_col_name,drug_col_name,dosage_col_name,response_col_name]]

    df_cl=cell_line_metadata[[ cl_colname_2b_mapped, 'name']]
    df_dr=drug_metadata[[ dr_colname_2b_mapped, 'name']]

    df_cl['name']=df_cl['name'].astype(str)
    
    if (dataname=="GDSC"):
        df_dr2=drug_metadata[[ dr_colname_2b_mapped_2, 'name']]
        df_dr2.columns=[dr_colname_2b_mapped,'name']
        df_dr=df_dr.append(df_dr2, ignore_index=True)
        
        df_cl=df_cl.fillna(0)
        #df_dr=df_dr.fillna(0)
        if(int_flag==0):
            df_cl[cl_colname_2b_mapped]=df_cl[cl_colname_2b_mapped].astype(int).astype(str)
            df_dr[dr_colname_2b_mapped]=df_dr[dr_colname_2b_mapped].astype(int).astype(str)
            
    
    dic_cl=dict(df_cl.values.tolist()) 
    dic_dr=dict(df_dr.values.tolist()) 

    return (dic_cl,dic_dr)


# In[ ]:

def getDictionary2(dataname):
    # MAP TO STANDARD NAMES
    
    metadata_filename="../data/metadata_2019_08_15_aanchal.xlsx"
    
    if(dataname=='CCLE'):
        cl_colname_2b_mapped=dataname+' name'
        dr_colname_2b_mapped=dataname+' name'
        drug_col_name='Compound' #
        cl_col_name='CCLE Cell Line Name' #
        dosage_col_name='Doses (uM)'
        response_col_name='Activity Data (median)'
    else:
        cl_colname_2b_mapped=dataname+' id'
        dr_colname_2b_mapped=dataname+' id (1)'
        dr_colname_2b_mapped_2=dataname+' id (2)'
        drug_col_name='DRUG_ID'#'Compound' #
        cl_col_name='COSMIC_ID'#'CCLE Cell Line Name' #
        dosage_col_name='dose'
        response_col_name='response'
        
    cell_line_metadata =pd.read_excel(metadata_filename,sheetname="cell line")
    drug_metadata =pd.read_excel(metadata_filename,sheetname="drug")

    raw=pd.read_csv("../data/drug_response/"+dataname+"/"+dataname+"_dose_response_scores.tsv", sep="\t")
    raw=raw[[cl_col_name,drug_col_name,dosage_col_name,response_col_name]]

    df_cl=cell_line_metadata[[ cl_colname_2b_mapped, 'name']]
    df_dr=drug_metadata[[ dr_colname_2b_mapped, 'name']]

    df_cl['name']=df_cl['name'].astype(str)
    
    if (dataname=="GDSC"):
        df_dr2=drug_metadata[[ dr_colname_2b_mapped_2, 'name']]
        df_dr2.columns=[dr_colname_2b_mapped,'name']
        df_dr=df_dr.append(df_dr2, ignore_index=True)
        
        df_cl=df_cl.fillna(0)
        df_cl[cl_colname_2b_mapped]=df_cl[cl_colname_2b_mapped].astype(int).astype(str)
    
    dic_cl=dict(df_cl.values.tolist()) 
    dic_dr=dict(df_dr.values.tolist()) 

    return (dic_cl,dic_dr)



import numpy as np
import scipy.stats.kde as kde

def hpd_grid(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode

    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results

    Returns
    ----------
    hpd: array with the lower 
          
    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes


# In[ ]:


def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """
    #print(x)
    x=np.array( sorted(x) )######i added
    #print(x)
    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_max- hdi_min, hdi_min, hdi_max#(hdi_max-hdi_min) #



def getDataObject(dataname,raw_filename,cl_col_name,drug_col_name,dosage_col_name,response_col_name):
    ''' get dataset objects'''
    raw=pd.read_csv(raw_filename, sep="\t")

    cl_list = sorted(list(set(raw[cl_col_name])))
    dr_list = sorted(list(set(raw[drug_col_name]))) 

    #cl_list=cl_list[:5]
    
    t0=pd.DataFrame()
    t=pd.DataFrame()
    t2=pd.DataFrame()

    for cl_name in cl_list:
        cl_df = raw[raw[cl_col_name] == cl_name] #has all drug profiling results for 1 particular cell-line 'cl_name' 

        for index, row in cl_df.iterrows():
            dosages = np.array(row[dosage_col_name].split(','), dtype=float)
            log2_dosage = np.log2(dosages)
            n_dosages = len(dosages)


            responses = np.array(row[response_col_name].split(','), dtype=float)

            t0 =  t0.append( pd.DataFrame([row[cl_col_name] ,row[drug_col_name] ]).transpose(), ignore_index=True)
            t  =   t.append([responses.T], ignore_index=True)
            t2=    t2.append([log2_dosage.T], ignore_index=True)

    t0.columns=['Cell_line_name','Drug _Name']#[cl_col_name,drug_col_name] 

    return t,t0,t2

def getHDI(x, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    '''if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:'''
    # Sort univariate node
    sx = np.sort(x)
    return np.array(calc_min_interval(sx, alpha))


from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #print(unique_labels(y_true, y_pred))
    classes = ['Resistant','Sensitive','Hump','Rising']#classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='CCLE',
           xlabel='GDSC')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

from itertools import chain
def chainer(s):
    return list(chain.from_iterable(s.str.split(',')))

def remove5dosageDrugs(raw, cl_col_name,drug_col_name,dosage_col_name,response_col_name):
    dosage_len_list=[]
    cols=raw.columns
    for i in range(len(raw[dosage_col_name]) ):
        dosage_len_list.append( len( np.array(raw.loc[i,dosage_col_name].split(','), dtype=float) ) )
    raw=raw.assign(dosages_length =  dosage_len_list)
    raw=raw[raw['dosages_length']==9]
    raw=raw[[cl_col_name,drug_col_name,dosage_col_name,response_col_name]] # length 190921 (33589 rows removed), 226 drugs (39 drugs removed)
    raw=pd.DataFrame(raw.values, columns=cols)
    return (raw)



############### similaritymeasures.py file from similaritymeasures package ##########3


import numpy as np
from scipy.spatial import distance
from scipy.spatial import minkowski_distance

# MIT License
#
# Copyright (c) 2018,2019 Charles Jekel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def poly_area(x, y):
    r"""
    A function that computes the polygonal area via the shoelace formula.

    This function allows you to take the area of any polygon that is not self
    intersecting. This is also known as Gauss's area formula. See
    https://en.wikipedia.org/wiki/Shoelace_formula

    Parameters
    ----------
    x : ndarray (1-D)
        the x locations of a polygon
    y : ndarray (1-D)
        the y locations of the polygon

    Returns
    -------
    area : float
        the calculated polygonal area

    Notes
    -----
    The x and y locations need to be ordered such that the first vertex
    of the polynomial correspounds to x[0] and y[0], the second vertex
    x[1] and y[1] and so forth


    Thanks to Mahdi for this one line code
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def is_simple_quad(ab, bc, cd, da):
    r"""
    Returns True if a quadrilateral is simple

    This function performs cross products at the vertices of a quadrilateral.
    It is possible to use the results to decide whether a quadrilateral is
    simple or complex. This function returns True if the quadrilateral is
    simple, and False if the quadrilateral is complex. A complex quadrilateral
    is a self-intersecting quadrilateral.

    Parameters
    ----------
    ab : array_like
        [x, y] location of the first vertex
    bc : array_like
        [x, y] location of the second vertex
    cd : array_like
        [x, y] location of the third vertex
    da : array_like
        [x, y] location of the fourth vertex

    Returns
    -------
    simple : bool
        True if quadrilateral is simple, False if complex
    """
    #   Compute all four cross products
    temp0 = np.cross(ab, bc)
    temp1 = np.cross(bc, cd)
    temp2 = np.cross(cd, da)
    temp3 = np.cross(da, ab)
    cross = np.array([temp0, temp1, temp2, temp3])
    #   See that cross products are greater than or equal to zero
    crossTF = cross >= 0
    #   if the cross products are majority false, re compute the cross products
    #   Because they don't necessarily need to lie in the same 'Z' direction
    if sum(crossTF) <= 1:
        crossTF = cross <= 0
    if sum(crossTF) > 2: #### ==2
        return True
    else:
        return False


def makeQuad(x, y):
    r"""
    Calculate the area from the x and y locations of a quadrilateral

    This function first constructs a simple quadrilateral from the x and y
    locations of the vertices of the quadrilateral. The function then
    calculates the shoelace area of the simple quadrilateral.

    Parameters
    ----------
    x : array_like
        the x locations of a quadrilateral
    y : array_like
        the y locations of a quadrilateral

    Returns
    -------
    area : float
        Area of quadrilateral via shoelace formula

    Notes
    -----
    This function rearranges the vertices of a quadrilateral until the
    quadrilateral is "Simple" (meaning non-complex). Once a simple
    quadrilateral is found, the area of the quadrilateral is calculated
    using the shoelace formula.
    """

    # check to see if the provide point order is valid
    # I need to get the values of the cross products of ABxBC, BCxCD, CDxDA,
    # DAxAB, thus I need to create the following arrays AB, BC, CD, DA

    AB = [x[1]-x[0], y[1]-y[0]]
    BC = [x[2]-x[1], y[2]-y[1]]
    CD = [x[3]-x[2], y[3]-y[2]]
    DA = [x[0]-x[3], y[0]-y[3]]

    isQuad = is_simple_quad(AB, BC, CD, DA)

    if isQuad is False:
        # attempt to rearrange the first two points
        x[1], x[0] = x[0], x[1]
        y[1], y[0] = y[0], y[1]
        AB = [x[1]-x[0], y[1]-y[0]]
        BC = [x[2]-x[1], y[2]-y[1]]
        CD = [x[3]-x[2], y[3]-y[2]]
        DA = [x[0]-x[3], y[0]-y[3]]

        isQuad = is_simple_quad(AB, BC, CD, DA)

        if isQuad is False:
            # place the second and first points back where they were, and
            # swap the second and third points
            x[2], x[0], x[1] = x[0], x[1], x[2]
            y[2], y[0], y[1] = y[0], y[1], y[2]
            AB = [x[1]-x[0], y[1]-y[0]]
            BC = [x[2]-x[1], y[2]-y[1]]
            CD = [x[3]-x[2], y[3]-y[2]]
            DA = [x[0]-x[3], y[0]-y[3]]

            isQuad = is_simple_quad(AB, BC, CD, DA)

    # calculate the area via shoelace formula
    area = poly_area(x, y)
    return area


def get_arc_length(dataset):
    r"""
    Obtain arc length distances between every point in 2-D space

    Obtains the total arc length of a curve in 2-D space (a curve of x and y)
    as well as the arc lengths between each two consecutive data points of the
    curve.

    Parameters
    ----------
    dataset : ndarray (2-D)
        The dataset of the curve in 2-D space.

    Returns
    -------
    arcLength : float
        The sum of all consecutive arc lengths
    arcLengths : array_like
        A list of the arc lengths between data points

    Notes
    -----
    Your x locations of data points should be dataset[:, 0], and the y
    locations of the data points should be dataset[:, 1]
    """
    #   split the dataset into two discrete datasets, each of length m-1
    m = len(dataset)
    a = dataset[0:m-1, :]
    b = dataset[1:m, :]
    #   use scipy.spatial to compute the euclidean distance
    dataDistance = distance.cdist(a, b, 'euclidean')
    #   this returns a matrix of the euclidean distance between all points
    #   the arc length is simply the sum of the diagonal of this matrix
    arcLengths = np.diagonal(dataDistance)
    arcLength = sum(arcLengths)
    return arcLength, arcLengths


def area_between_two_curves(exp_data, num_data):
    r"""
    Calculates the area between two curves.

    This calculates the area according to the algorithm in [1]_. Each curve is
    constructed from discretized data points in 2-D space, e.g. each curve
    consists of x and y data points.

    Parameters
    ----------
    exp_data : ndarray (2-D)
        Curve from your experimental data.
    num_data : ndarray (2-D)
        Curve from your numerical data.

    Returns
    -------
    area : float
        The area between exp_data and num_data curves.

    References
    ----------
    .. [1] Jekel, C. F., Venter, G., Venter, M. P., Stander, N., & Haftka, R.
        T. (2018). Similarity measures for identifying material parameters from
        hysteresis loops using inverse analysis. International Journal of
        Material Forming. https://doi.org/10.1007/s12289-018-1421-8

    Notes
    -----
    Your x locations of data points should be exp_data[:, 0], and the y
    locations of the data points should be exp_data[:, 1]. Same for num_data.
    """
    # Calculate the area between two curves using quadrilaterals
    # Consider the test data to be data from an experimental test as exp_data
    # Consider the computer simulation (results from numerical model) to be
    # num_data
    #
    # Example on formatting the test and history data:
    # Curve1 = [xi1, eta1]
    # Curve2 = [xi2, eta2]
    # exp_data = np.zeros([len(xi1), 2])
    # num_data = np.zeros([len(xi2), 2])
    # exp_data[:,0] = xi1
    # exp_data[:,1] = eta1
    # num_data[:, 0] = xi2
    # num_data[:, 1] = eta2
    #
    # then you can calculate the area as
    # area = area_between_two_curves(exp_data, num_data)

    n_exp = len(exp_data)
    n_num = len(num_data)

    # the length of exp_data must be larger than the length of num_data
    if n_exp < n_num:
        temp = num_data.copy()
        num_data = exp_data.copy()
        exp_data = temp.copy()
        n_exp = len(exp_data)
        n_num = len(num_data)

    # get the arc length data of the curves
    # arcexp_data, _ = get_arc_length(exp_data)
    _, arcsnum_data = get_arc_length(num_data)

    # let's find the largest gap between point the num_data, and then
    # linearally interpolate between these points such that the num_data
    # becomes the same length as the exp_data
    for i in range(0, n_exp-n_num):
        a = num_data[0:n_num-1, 0]
        b = num_data[1:n_num, 0]
        nIndex = np.argmax(arcsnum_data)
        newX = (b[nIndex] + a[nIndex])/2.0
        #   the interpolation model messes up if x2 < x1 so we do a quick check
        if a[nIndex] < b[nIndex]:
            newY = np.interp(newX, [a[nIndex], b[nIndex]],
                             [num_data[nIndex, 1], num_data[nIndex+1, 1]])
        else:
            newY = np.interp(newX, [b[nIndex], a[nIndex]],
                             [num_data[nIndex+1, 1], num_data[nIndex, 1]])
        num_data = np.insert(num_data, nIndex+1, newX, axis=0)
        num_data[nIndex+1, 1] = newY

        _, arcsnum_data = get_arc_length(num_data)
        n_num = len(num_data)

    # Calculate the quadrilateral area, by looping through all of the quads
    area = []
    for i in range(1, n_exp):
        tempX = [exp_data[i-1, 0], exp_data[i, 0], num_data[i, 0],
                 num_data[i-1, 0]]
        tempY = [exp_data[i-1, 1], exp_data[i, 1], num_data[i, 1],
                 num_data[i-1, 1]]
        area.append(makeQuad(tempX, tempY))
    return np.sum(area)



