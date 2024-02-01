import numpy as np
#from src.uframe import uframe_instance
import scipy
from .uframe_instance import uframe_instance
from scipy import stats
from sklearn.neighbors import KernelDensity
import sklearn.neighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import miceforest as mf
import math
import pickle 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
from itertools import compress
from .helper import analysis_table, analysis_table_distr, plot_pca, KL 
from typing import Optional, List, Dict


#SOLANGE DAS NICHT FUNKTIONIERT; MÜSSEN WERTE IN KATEGORIELLEN SPALTEN GANZZAHLIG SEIN
#nur Integer als Keys für kategorielle Verteilung zugelassen, muss append Funktion entsprechend anpassen 
#checke jeweils, ob schon cat value dict vorhanden, falls ja, ob es ergänzt werden muss
#falls kein cat value dict vorhanden, suche eine cat value Liste, die geeignet ist 

#expected value im kategoriellen Fall und geeignete Anpassung der get dummies Funktionen
#nach ev darf mit diesme Array einfach kein get dummies with array gecallt werden 

#self cat value dicts muss automatisch erstellt und geupdatet werden, damit ev so funktioniert 
#kann dann auch die OHE Funktionen damit anpassen 


class uframe():
    """
    A class used for storing and working with uncertain data.

    ...
    
    Parameters
    ----------
    data : list
        A list of data instances of class uframe_instance
    columns: list
        List of column names
    rows: list
        List of row names
    _colnames: dict
        A dictionary with column names as keys and the corresponding indices in
        the data as values
    _rownames: dict
        A dictionary with row names as keys and the corresponding indices in
        the data as values
    col_dtype: list
        A list of the datatypes of the columns
    _col_dtype: dict
        A dictionary with column names as keys and the corresponding data
        types as values

    Methods
    -------
    modal()
        Returns a numpy array of the data, where all uncertain values are
        replaced with their modal values.

    sample(n = 1, seed = None)
        Samples n samples for each instance and returns one numpy array,
        where n samples from each instance are below each other.

    ev()
        Returns a numpy array where all uncertain values are replaced with
        their expected values

    get_dummies()
        Performs one hot encoding on all categorical columns in the data.
    
    var

    update():
        Update specific instance, not immediately relevant

    append(new = None):
        Append new rows to the data which can be either certain or uncertain.
        In place. New rows have to be numpy arrays at first, later add
        different sources of uncertain new rows.


    """

    def __init__(self,  new = None, colnames = None, rownames = None):

        self.data = []
        self._columns = []
        self._colnames = {}
        self._rows = []
        self._rownames = {}
        self._col_dtype = []

        
        if new is not None: 
            self.append(new = new, colnames = colnames, rownames = rownames)

            
            
    def analysis_frame(self, true_data, save = False, **kwargs): 
        df = pd.DataFrame(data= np.nan, 
                          columns = self._columns,
                          index = pd.Index(["changed",
                                            "changed_mode_MAE","changed_mode_RMSE",
                                            "changed_mode_VAR","changed_mode_dif_quantile",
                                            "overall_mode_MAE","overall_mode_RMSE",
                                            "overall_mode_VAR","overall_mode_dif_quantile",
                                            "changed_ev_MAE","changed_ev_RMSE",
                                            "changed_ev_VAR","changed_ev_dif_quantile",
                                            "overall_ev_MAE","overall_ev_RMSE",
                                            "overall_ev_VAR","overall_ev_dif_quantile"]))
        
        
        for i in range(len(df.columns)):
            changed = self.mode()[:,i] != true_data[:,i]
            df.iloc[0,i] = sum(changed)/len(self)
            
            tab = analysis_table(true_data[changed,i], self.mode()[changed,i])
            df.iloc[1:5,i] = [row[1] for row in tab]
            
            tab = analysis_table(true_data[:,i], self.mode()[:,i])
            df.iloc[5:9,i] = [row[1] for row in tab]
            
            tab = analysis_table(true_data[changed,i], self.ev()[changed,i])
            df.iloc[9:13,i] = [row[1] for row in tab]
            
            tab = analysis_table(true_data[:,i], self.ev()[:,i])
            df.iloc[13:17,i] = [row[1] for row in tab]
            
        if save is not False:
              if not type (save) == str:
                  save  = 'analysis_uframe.csv'
              else: 
                  save = save + ".csv"
            
              df.to_csv(save, index=True)
            
        return df
   
    def analysis(self, true_data, save = False, **kwargs): 
        
        _save = False
        if save is not False:
              if not type (save) == str:
                  save  = 'analysis_uframe'
              
              pdf = PdfPages(str(save)+'.pdf') 
              _save = True 
        
              
          
          #Mode analysis
        mode = self.mode()
        ev = self.ev()
        
        
        cont = [k == 'continuous' for k in self._col_dtype]
            
   
        

        for i in range(len(self.columns)):
            if self._col_dtype[i]== 'continuous':
                #mode
                hist_fig, axs = plt.subplots(4, 2, figsize=(11.69,8.27))
                
                changed = ev[:,i].round(4) != true_data[:,i].round(4)

                #links
                axs[0,0].set_title('Mode, Variable: '+ str(self.columns[i]))
                axs[0,0].hist(mode[changed,i], **kwargs)
                axs[1,0].axis('tight')
                axs[1,0].axis('off')
                axs[1,0].table(analysis_table_distr(mode[changed,i],true_data[changed,i],"(Mode|True)"), loc = "center")
                
                
                #rechts
                axs[0,1].set_title('True values, Variable:'+str(self.columns[i]))
                axs[0,1].hist(true_data[changed,i], **kwargs)
                axs[1,1].axis('tight')
                axs[1,1].axis('off')
                axs[1,1].table(analysis_table_distr(true_data[changed,i],mode[changed,i],"(True|Mode)"), loc = "center")
                        

                #EV analysis
                changed = ev[:,i].round(4) != true_data[:,i].round(4)

                axs[2,0].set_title('EV, Variable:'+ str(self.columns[i]))
                axs[2,0].hist(ev[changed ,i], **kwargs)
                axs[3,0].axis('tight')
                axs[3,0].axis('off')
                axs[3,0].table(analysis_table_distr(ev[changed,i],true_data[changed,i],"(EV|True)"), loc = "center")

                
                axs[2,1].set_title('True values, Variable:'+str(self.columns[i]))
                axs[2,1].hist(true_data[changed ,i], **kwargs)
                axs[3,1].axis('tight')
                axs[3,1].axis('off')
                axs[3,1].table(analysis_table_distr(true_data[changed,i],ev[changed,i],"(True|EV)"), loc = "center")
                
                if _save == True: 
                    hist_fig.savefig(pdf, format = 'pdf')        
                   
                    
                hist_fig, axs = plt.subplots(4, 2, figsize=(11.69,8.27))

                    
                #analysis of variances 
                var= self.var(n = 100)
                axs[0,0].set_title('Variances, Variable: '+ str(self.columns[i]))
                axs[0,0].hist(var[changed,i], **kwargs)
                axs[0,1].axis('tight')
                axs[0,1].axis('off')
                axs[0,1].table([["Mean",str(round(np.mean(var[changed,i]),2))],
                               ["Var",str(round(np.var(var[changed,i]),2))]], loc = "center")
                

                mae_mode_to_m = np.mean(abs(mode[changed,i]-true_data[changed,i] ))/np.mean(true_data[changed,i])
                mae_ev_to_m = np.mean(abs(ev[changed,i]-true_data[changed,i] ))/np.mean(true_data[changed,i])
                var_to_m = np.mean(var[changed,i])/np.mean(true_data[changed,i])
                
                axs[1,0].axis('tight')
                axs[1,0].axis('off')
                other_measures = axs[1,0].table([["Ratios of Variable:"+str(self.columns[i]),""],
                                ["MAE_EV/mean_true", str(round(mae_ev_to_m ,2))],
                                ["MAE_Mode/mean_true", str(round(mae_mode_to_m ,2))],
                                ["mean_var/mean_true", str(round(var_to_m,2))],
                                ["Uncertain Values/Total",str(sum(changed)) +"/"+str(len(changed))+ ", "+str(round(100*sum(changed)/len(changed),1))+"%"]],
                               loc = "center") 

                axs[1,1].axis('tight')
                axs[1,1].axis('off')
                
                other_measures.auto_set_font_size(False)
                other_measures.set_fontsize(12)
                other_measures.auto_set_column_width([0,1])
                
                
                changed = mode[:,i] != true_data[:,i]
                
                #Residues Mode
                axs[2,0].set_title('Residue of Mode, Variable: '+ str(self.columns[i]))
                axs[2,0].hist(mode[changed,i]-true_data[changed,i])
                axs[3,0].axis('tight')
                axs[3,0].axis('off')
                axs[3,0].table(analysis_table(true_data[changed,i], mode[changed,i]), loc = "center")
                
                
                #Residues EV
                changed = ev[:,i].round(4) != true_data[:,i].round(4)
                axs[2,1].set_title('Residue of EV, var:'+ str(self.columns[i]))
                residues = ev[changed,i]-true_data[changed,i]    
                axs[2,1].hist(residues, **kwargs)
                axs[3,1].axis('tight')
                axs[3,1].axis('off')
                axs[3,1].table(analysis_table(true_data[changed,i], ev[changed,i]), loc = "center")
                
                plt.show()
                if _save == True: 
                    hist_fig.savefig(pdf, format = 'pdf')        
                       
                    
                #Comparison to undeleted data to find potentioal Biases 
                
                hist_fig, axs = plt.subplots(3, 2, figsize=(11.69,8.27), constrained_layout=True)
                """
                axs[0,0].set_title('EV, Variable: '+ str(self.columns[i]))
                axs[0,0].hist(ev[changed,i])
                axs[0,1].set_title('Mode, Variable: '+ str(self.columns[i]))
                axs[0,1].hist(mode[changed,i])
                """
                axs[0,0].set_title('True Values, ucertain, Variable: '+ str(self.columns[i]))
                axs[0,0].hist(true_data[changed,i])
                axs[0,1].set_title('True Values, certain, Variable: '+ str(self.columns[i]))
                axs[0,1].hist(true_data[~changed,i])
                
                
                bins = np.linspace(min(true_data[:,i]), max(true_data[:,i]), 30)
                axs[1,0].set_title('Comparison, certain & uncertain, Variable: '+ str(self.columns[i]))
                axs[1,0].hist([true_data[~changed,i], true_data[changed,i]], bins, label=['certain instances', 'uncertain instances'])
                axs[1,0].legend(loc='upper right', fontsize = "5")
                
                sns.kdeplot(mode[changed,i], bw_adjust=0.5, label='Mode', ax = axs[1,1])
                sns.kdeplot(ev[changed,i], bw_adjust=0.5, label='EV', ax = axs[1,1])
                sns.kdeplot(true_data[~changed,i], bw_adjust=0.5, label='True certain', ax = axs[1,1])
                sns.kdeplot(true_data[changed,i], bw_adjust=0.5, label='True uncertain', ax = axs[1,1])
                axs[1,1].set_title('Comparison of KDE\'s '+ str(self.columns[i]))
                axs[1,1].set_xlabel('Value')
                axs[1,1].legend(loc='upper right', fontsize = "5")
                
                axs[2,0].axis('tight')
                axs[2,0].axis('off')
                try: 
                    ks_statistic, p_value = scipy.stats.ks_2samp(true_data[changed,i], true_data[~changed,i])
                    p_value = str(round(p_value,4))
                except: 
                    ks_statistic, p_value = "NA", "NA"

                tab1= axs[2,0].table([["KS (True Certain | True Uncertain)", p_value],
                                ["KL(True Certain|True Uncertain)",str(KL(true_data[~changed,i],true_data[changed,i]))],
                                ["KL(Mode|True Uncertain)",str(KL(mode[changed,i],true_data[changed,i]))],
                                ["KL(EV|True Uncertain)",str(KL(ev[changed,i],true_data[changed,i]))]], loc = "center")
                

                axs[2,1].axis('tight')
                axs[2,1].axis('off')
                
                tab1.auto_set_font_size(False)
                tab1.set_fontsize(14)
                tab1.auto_set_column_width([0,1])
                
                tab = axs[2,1].table([["Mean, Mode uncertain", str(round(np.mean(mode[changed,i]),2))],
                                ["Mean, EV uncertain", str(round(np.mean(ev[changed,i]),2))],
                                ["Mean, true uncertain ", str(round(np.mean(true_data[changed,i]),2))],
                                ["Mean, true certain",str(round(np.mean(true_data[~changed,i]),2))],
                                ["Mean, true certain & mode uncertain", str(round(np.mean(mode[:,i]),2))],
                                ["Mean, true certain & EV uncertain", str(round(np.mean(ev[:,i]),2))],
                                
                                ["Var, Mode uncertain", str(round(np.var(mode[changed,i]),2))],
                                ["Var, EV uncertain", str(round(np.var(ev[changed,i]),2))],
                                ["Var, true uncertain", str(round(np.var(true_data[changed,i]),2))],
                                ["Var, true certain", str(round(np.var(true_data[~changed,i]),2))],
                                ["Var, true certain & mode uncertain",str(round(np.var(mode[:,i]),2))],
                                ["Var, true certain & ev uncertain",str(round(np.var(ev[:,i]),2))]
                                ], loc = "center")
                #axs[3,1].table.auto_set_font_size(False)
                tab.auto_set_font_size(False)
                tab.set_fontsize(14)
                tab.auto_set_column_width([0,1])
                plt.show()
                if _save == True: 
                    hist_fig.savefig(pdf, format = 'pdf')        
                
                
                
            #Accuracy in case of categorical
        """
        if self._col_dtype[i]== 'categorical':
            hist_fig, hist_ax = plt.subplots(1, 1, figsize=(9, 3))
            hist_ax.set_title('Accuracy of Mode, var:'+ str(self._colnames[i]))
                
            hist_ax.hist(mode[:,i] == true_data[:,i],**kwargs)
              
            if _save == True: 
                hist_fig.savefig(pdf, format = 'pdf')        
        """
        
        if 'continuous' in self._col_dtype: 
            colnames= []
            for k in range(len(self.columns)):
                colnames.append(self.columns[k])
            
            
            fig, axs = plt.subplots(4, 1, figsize=(11.69,8.27))
            
            sns.boxplot(pd.DataFrame(mode[:,cont], columns = list(compress(colnames,cont))), ax = axs[0]).set_title("Mode Distributions")
            sns.boxplot(pd.DataFrame(ev[:,cont], columns = list(compress(colnames,cont))), ax = axs[1]).set_title("EV Distributions")
            
            
            
            res_mode = abs(mode - true_data).mean(axis = 0)
            res_ev = abs(ev - true_data).mean(axis= 0)
            
            axs[2].bar([str(k) for k in list(compress(colnames,cont))], res_mode )
            for i in range(len(list(compress(colnames,cont)))):
                axs[2].text(i,round(res_mode[i],2),round(res_mode[i],2))
            axs[2].set_title("Mode MAE per Variable")
       
            axs[3].bar([str(k) for k in list(compress(colnames,cont))], res_ev)
            for i in range(len(list(compress(colnames,cont)))):
                axs[3].text(i,round(res_ev[i],2),round(res_ev[i],2))
            axs[3].set_title("EV MAE per Variable")
            
            plt.show()
            if _save == True: 
                fig.savefig(pdf, format = 'pdf')
            
                
            fig, axs = plt.subplots(2, 1, figsize=(11.69,8.27))
            dist_mode = np.linalg.norm(mode - true_data, axis = 1)
            dist_ev = np.linalg.norm(ev - true_data, axis = 1)
            
            sns.boxplot(pd.DataFrame(np.array([dist_mode,dist_ev]).transpose(), columns = ["Mode", "ev"]), ax = axs[0]).set_title("Euclidean Distances to true value")
            
            plot_pca(true_data, axs = axs[1], uncertain = changed)
            
            plt.show()
            if _save == True: 
                fig.savefig(pdf, format = 'pdf')
        
        pdf.close()
        plt.close()

 
    def append(self, new=None, colnames=None, rownames=None):

        if new is None:
            return

        if isinstance(new, np.ndarray):
            self.append_from_numpy(new, colnames, rownames)

        elif isinstance(new, scipy.stats._kde.gaussian_kde):
            self.append_from_scipy_kde([new], colnames)

        elif isinstance(new, sklearn.neighbors._kde.KernelDensity):
            self.append_from_sklearn_kde([new], colnames)

        elif isinstance(new, uframe_instance):
            self.append_from_uframe_instance([new], colnames)

        # one categorical distribution
        elif isinstance(new, dict):
            self.append_from_categorical([new], colnames)

        elif issubclass(type(new), scipy.stats.rv_continuous) or issubclass(type(new),
                                                                            scipy.stats._distn_infrastructure.rv_continuous_frozen):
            self.append_from_rv_cont([new], colnames)

        elif issubclass(type(new), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(new),
                                                                                             scipy.stats._multivariate.multi_rv_frozen):
            self.append_from_rv_cont([new], colnames)

        elif isinstance(new, list):

            if len(new) == 4 and isinstance(new[0], np.ndarray) and isinstance(new[1], dict):
                #try:
                self.append_with_mixed_distributions(certain=new[0], continuous=new[1],
                                                         categorical=new[2], indices=new[3])
                #except BaseException:
                    #raise ValueError("Unsuitable parameter new given")

            if len(new) == 2 and isinstance(new[0], np.ndarray) and isinstance(new[1], dict):

                # check if dict contains continuous or categorical distributions
                if isinstance(new[1][list(new[1].keys())[0]], list) and isinstance(new[1][list(new[1].keys())[0]][0], dict):
                    self.append_with_mixed_categorical(new[0], new[1], colnames)
                else:
                    self.append_from_mix_with_distr(new[0], new[1], colnames)

            # one with two lists (or a list of lists of length 2) with only continuous und categorical variables

            if isinstance(new[0], scipy.stats._kde.gaussian_kde):
                self.append_from_scipy_kde(new, colnames)
            elif isinstance(new[0], sklearn.neighbors._kde.KernelDensity):
                self.append_from_sklearn_kde(new, colnames)
            elif isinstance(new[0], uframe_instance):
                #print("Entered uframe instance list append loop")
                self.append_from_uframe_instance(new, colnames)
            elif issubclass(type(new[0]), scipy.stats.rv_continuous) or issubclass(type(new[0]), scipy.stats._distn_infrastructure.rv_continuous_frozen):
                self.append_from_rv_cont(new, colnames)
            elif issubclass(type(new[0]), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(new[0]),
                                                                                                    scipy.stats._multivariate.multi_rv_frozen):
                self.append_from_rv_cont(new, colnames)
            # Fall, wo Liste von Listen von 1D Distribs übergeben wird
            # checke, dass hier keine kategorischen Verteilungen übergeben werden
            elif isinstance(new[0], list):
                if len(new[0]) > 0 and not isinstance(new[0][0], dict):
                    self.append_from_rv_cont(new, colnames)
                elif len(new[0]) > 0 and isinstance(new[0][0], dict):
                    self.append_from_categorical(new, colnames)
        
        else: 
            raise ValueError("Unknown type of new")
            
        # sollte evtl. lieber die rownames in jeder append Funktion gesondert adressieren
        if isinstance(new, np.ndarray):
            return
        elif not isinstance(new, list):
            self.addRownames([new], rownames)
        elif isinstance(new[0], np.ndarray):
            self.addRownames(new[0], rownames)
        else:
            self.addRownames(new, rownames)

        return

    # append a numpy array with certain data (2D-array)
    def append_from_numpy(self, new=None, colnames=None, rownames=None):
        
        self.append_with_mixed_distributions(certain=new, continuous={}, categorical={},
                                             indices={}, colnames=colnames)
        
        if rownames is not None:

            # check rownames for duplicates (also with previously existing rownames)
            if len(set(self._rows + rownames)) != len(self.rows) + len(rownames):
                raise ValueError("Duplicates among rownames")
            if len(rownames) != len(new):
                raise ValueError(
                    "Number of rownames given does not match number of rows given")

            self._rows = self._rows + rownames
            self._rownames.update(
                {rownames[i]: len(self.data) - len(new) + i for i in range(len(rownames))})
        else:

            self._rows = self.rows + [*list(range(len(self.data) - len(new), len(self.data)))]
            # print("Rows", self.rows, "updated with", [*list(range(len(self.data) - len(new), len(self.data)))])
            self._rownames.update({i: i for i in range(
                len(self.data) - len(new), len(self.data))})
       
        return

    # distr list: list of list of dicts (1 list of dicts per instance)
    def append_from_categorical(self, distr_list, colnames=None):
        
        #need correct dimension from first element of distr_list
        dimension = len(distr_list[0])
        a = np.empty((len(distr_list),dimension))
        a[:]= np.nan
        self.append_with_mixed_distributions(certain=a, continuous={}, 
                                             categorical={i:distr_list[i] for i in range(len(distr_list))},
                                             indices={i:[[], [*list(range(dimension))]] for i in range(len(distr_list))})
        '''
        if len(self.columns) > 0:
            dimensions = len([len(l) for l in distr_list if len(l) == len(self.columns)])
            if dimensions != len(distr_list):
                raise ValueError("Dimension of list element does not match dimension of uframe")
            conts = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'continuous']
            if len(conts) > 0:
                raise ValueError("Categorical distribution in continuous column")
        else:
            dimensions = len([len(l) for l in distr_list if len(l) == len(distr_list[0])])
            if dimensions != len(distr_list):
                raise ValueError("Distributions in list must have same dimension")

            if colnames is None:
                self._columns = [*list(range(len(distr_list[0])))]
                self._colnames = {i: i for i in range(len(distr_list[0]))}
            else:
                if len(colnames) != len(distr_list[0]):
                    raise ValueError("Length of column list does not match dim. of distributions")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i, name in enumerate(colnames)}
            self._col_dtype = len(self.columns) * ['categorical']

        for i, distr in enumerate(distr_list):
            self.data.append(uframe_instance(continuous=None, categorical=distr,
                                             certain_data=None,
                                             indices=[[], [], [*list(range(len(distr)))]]))

        return
    
    '''
    def append_from_scipy_kde(self, kernel_list, colnames=None):
        
        dimension = kernel_list[0].d
        a = np.empty((len(kernel_list), dimension))
        a[:]= np.nan
        self.append_with_mixed_distributions(certain=a, continuous={i:kernel_list[i] for i in range(len(kernel_list))}, 
                                             categorical={}, 
                                             indices={i:[[*list(range(dimension))],[]] for i in range(len(kernel_list))})

        '''
        if len(self.columns) > 0:

            dimensions = len(
                [kernel.d for kernel in kernel_list if kernel.d == len(self.columns)])
            if dimensions != len(kernel_list):
                raise ValueError(
                    "Dimension of list element does not match dimension of uframe")
            categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
            if len(categoricals) > 0:
                raise ValueError("Continuous kernel in categorical column")

        else:
            # check if all kernels have the same dimension
            dimensions = len(
                [kernel.d for kernel in kernel_list if kernel.d == kernel_list[0].d])
            if dimensions != len(kernel_list):
                raise ValueError("Kernels in list must have same dimension")

            if colnames is None:
                self._columns = [*list(range(kernel_list[0].d))]
                self._colnames = {i: i for i in range(kernel_list[0].d)}
            else:
                if len(colnames) != kernel_list[0].d:
                    raise ValueError(
                        "Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}

            self._col_dtype = len(self.columns) * ['continuous']

        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(continuous=kernel, certain_data=None,
                                             indices=[[], [*list(range(kernel.d))], []]))
         '''   
        return

    def append_from_sklearn_kde(self, kernel_list, colnames=None):
        
        dimension = kernel_list[0].n_features_in_
        a = np.empty((len(kernel_list), dimension))
        a[:]= np.nan
        self.append_with_mixed_distributions(certain=a, continuous={i:kernel_list[i] for i in range(len(kernel_list))}, 
                                             categorical={}, 
                                             indices={i:[[*list(range(dimension))],[]] for i in range(len(kernel_list))})
        
        '''
        if len(self.columns) > 0:

            dimensions = len(
                [kernel.n_features_in_ for kernel in kernel_list if kernel.n_features_in_ == len(self.columns)])
            if dimensions != len(kernel_list):
                raise ValueError(
                    "Dimension of list element does not match dimension of uframe")
            categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
            if len(categoricals) > 0:
                raise ValueError("Continuous kernel in categorical column")

        else:
            # check if all kernels have the same dimension
            dimensions = len(
                [kernel.n_features_in_ for kernel in kernel_list if kernel.n_features_in_ == kernel_list[0].d])
            if dimensions != len(kernel_list):
                raise ValueError("Kernels in list must have same dimension")

            if colnames is None:
                self._columns = [*list(range(kernel_list[0].n_features_in_))]
                self._colnames = {i: i for i in range(
                    kernel_list[0].n_features_in_)}
            else:
                if len(colnames) != kernel_list[0].n_features_in_:
                    raise ValueError(
                        "Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}
            self._col_dtype = len(self.columns) * ['continuous']

        for i, kernel in enumerate(kernel_list):
            self.data.append(uframe_instance(continuous=kernel, certain_data=None,
                                             indices=[[], [*list(range(kernel.n_features_in_))], []]))
        '''

    # jeder Eintrag von distr_list muss entweder eine multivariate Verteilung über alle Variablen
    # oder eine Liste unabhängiger 1D-Verteilungen
    def append_from_rv_cont(self, distr_list, colnames=None):
        
        distr= distr_list[0]
        if issubclass(type(distr), scipy.stats._multivariate.multi_rv_generic) or issubclass(type(distr), scipy.stats._multivariate.multi_rv_frozen):
            dimension = distr.mean.shape[0]
        elif issubclass(type(distr), scipy.stats._distn_infrastructure.rv_continuous_frozen) or issubclass(type(distr), scipy.stats.rv_continuous):
            dimension = distr.rvs(size=1).shape[0]
        elif isinstance(distr, list):
            dimension = sum([d.rvs(size=1).shape[0] for d in distr])
        else:
            raise ValueError("Invalid argument in distribution list")
        
        a = np.empty((len(distr_list),dimension))
        a[:] = np.nan
        self.append_with_mixed_distributions(certain=a, 
                                             continuous = {i:distr_list[i] for i in range(len(distr_list))}, 
                                             categorical={}, 
                                             indices={i:[[*list(range(dimension))],[]] for i in range(len(distr_list))})
            
        
        return

    # assume dictionary with indices of incomplete lines as keys and scipy kdes as values
    # nan values for uncertain values in array certain
    def append_from_mix_with_distr(self, certain, distr, colnames=None):
        
        self.append_with_mixed_distributions(certain=certain, 
                                             continuous=distr, 
                                             categorical={}, 
                                             indices={i:[list(np.where(np.isnan(certain[i]))[0]),[]] for i in distr.keys()})
        
        return

    def append_with_mixed_categorical(self, certain, distr, colnames=None):
        
        self.append_with_mixed_distributions(certain=certain, continuous={}, 
                                             categorical=distr, 
                                             indices= {i:[[],list(np.where(np.isnan(certain[i]))[0])] for i in distr.keys()})
        
        return

    def append_with_mixed_distributions(self, certain, continuous, categorical, indices, colnames=None):

        # certain is an array with missing values indicated by nan values
        # continuous: dict, keys are indices of lines with missing values, values are MV distr., kernels or lists of 1D distr.
        # categorical: dict, keys are indices of lines with missing values, values are lists of dicts with categorical distributions for them
        # indices: dict, keys are indices of lines with missing values, values are lists of 2 lists, first list are indices of missing values with continuous distr, second with categorical
        # indices have to match the missing values in certain and the dimensions of the continuous and categorical distr.

        #print("Indices", indices)
        
        if len(self.columns) > 0:

            if len(self.columns) != certain.shape[1]:
                raise ValueError("Dimensions of new data do not match uframe")
            conts = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'conts']
            for index in conts:
                if np.any(np.isnan(certain[:, index])):
                    raise ValueError("Categorical distribution in continuous column")
        else:
            if colnames is None:
                self._columns = [*list(range(certain.shape[1]))]
                self._colnames = {i: i for i in range(certain.shape[1])}
            else:
                if len(colnames) != certain.shape[1]:
                    raise ValueError(
                        "Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}

        self._col_dtype = []
        for col_index in range(len(self.columns)):
            if np.any(np.isnan(certain[:, col_index])):
                categoricals = [i for i in range(len(certain)) if i in indices.keys() and col_index in indices[i][1]]
                if len(categoricals) > 0:
                    conts = [i for i in range(len(certain)) if i in indices.keys() and col_index in indices[i][0]]
                    if len(conts) > 0:
                        raise ValueError("Continuous and categorical distr in same column")
                        certain_values = certain[:, col_index][np.isnan(certain[:, col_index] == False)]
                        if np.any(certain_values - np.floor(certain_values)):
                            raise ValueError("Float values and categorical distr in same column")
                    self._col_dtype.append('categorical')
                else:
                    self._col_dtype.append('continuous')
            else:
                self._col_dtype.append('continuous')

        # only for continuous case
        for i in range(len(certain)):
            if i not in continuous.keys():
                if i not in categorical.keys():
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns)
                    self.data.append(uframe_instance(continuous=None, categorical=None,
                                                     certain_data=certain[i, :],
                                                     indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                              [], []]))
                else:
                    assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(categorical[i]) == len(self.columns)
                    self.data.append(uframe_instance(continuous=None,
                                                     categorical=categorical[i],
                                                     certain_data=certain[i][np.isnan(
                                                         certain[i]) == False],
                                                     indices=[list(np.where(np.isnan(certain[i]) == False)[0]), [],
                                                              list(np.where(np.isnan(certain[i]))[0])]))

            else:
                if i not in categorical.keys():
                    if isinstance(continuous[i], scipy.stats._kde.gaussian_kde):

                        assert len(np.where(np.isnan(certain[i]) == False)[0]) + continuous[i].d == len(self.columns)
                    elif isinstance(continuous[i], sklearn.neighbors._kde.KernelDensity):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + continuous[i].n_features_in_ == len(self.columns)
                    elif isinstance(continuous[i], list):
                        distr_list = continuous[i]
                        # check if every list element is rv continuous or rv continuous frozen

                        assert len([l for l in distr_list if issubclass(type(l),
                                                                        scipy.stats.rv_continuous)
                                    or issubclass(type(l),
                                                  scipy.stats._distn_infrastructure.rv_continuous_frozen)]) == len(distr_list)

                        # check if length of list is correct
                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(continuous[i]) == len(self.columns)
                    elif issubclass(type(continuous[i]),
                                    scipy.stats._multivariate.multi_rv_generic) or issubclass(type(continuous[i]),
                                                                                              scipy.stats._multivariate.multi_rv_frozen):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + continuous[i].mean.shape[0] == len(self.columns)

                    # check if it is 1D distr (one missing attribute in this row)
                    elif issubclass(type(continuous[i]),
                                    scipy.stats.rv_continuous) or issubclass(type(continuous[i]),
                                                                             scipy.stats._distn_infrastructure.rv_continuous_frozen):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) == len(self.columns) - 1
                    else:
                        raise ValueError("Distribution type not supported")

                    self.data.append(uframe_instance(continuous=continuous[i],
                                                     certain_data=certain[i][np.isnan(
                                                         certain[i]) == False],
                                                     indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                              list(np.where(np.isnan(certain[i]))[0]), []]))
                if i in categorical.keys():

                    # check that the missing value indices and the distributions indices match
                    if set(np.where(np.isnan(certain[i, :]))[0]) != set(indices[i][0] + indices[i][1]):
                        raise ValueError("Indices of distributions do not match indices of missing values")
                    if isinstance(continuous[i], scipy.stats._kde.gaussian_kde):

                        assert len(np.where(np.isnan(certain[i]) == False)[0])
                        + continuous[i].d + len(categorical[i]) == len(self.columns)

                    elif isinstance(continuous[i], sklearn.neighbors._kde.KernelDensity):

                        left = len(list(np.where(np.isnan(certain[i]) == False)[0]))+ continuous[i].n_features_in_ + len(categorical[i])
                        right = len(self.columns)
                        assert left==right
                        
                    elif isinstance(continuous[i], list):
                        distr_list = continuous[i]
                       
                        assert len([l for l in distr_list if issubclass(type(l),
                                                                        scipy.stats.rv_continuous)
                                    or issubclass(type(l),
                                                  scipy.stats._distn_infrastructure.rv_continuous_frozen)]) == len(distr_list)

                        # check if length of list is correct
                        assert len(list(np.where(np.isnan(certain[i]) == False)[0]))
                        + len(continuous[i]) + len(categorical[i]) == len(self.columns)

                    elif issubclass(type(continuous[i]),
                                    scipy.stats._multivariate.multi_rv_generic) or issubclass(type(continuous[i]),
                                                                                              scipy.stats._multivariate.multi_rv_frozen):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0]))
                        + continuous[i].mean.shape[0] + len(categorical[i]) == len(self.columns)

                    elif issubclass(type(continuous[i]),
                                    scipy.stats.rv_continuous) or issubclass(type(continuous[i]),
                                                                             scipy.stats._distn_infrastructure.rv_continuous_frozen):

                        assert len(list(np.where(np.isnan(certain[i]) == False)[0])) + len(categorical[i]) == len(self.columns)

                    else:
                        raise ValueError("Distribution type not supported")

                    self.data.append(uframe_instance(continuous=continuous[i],
                                                     categorical=categorical[i],
                                                     certain_data=certain[i][np.isnan(
                                                         certain[i]) == False],
                                                     indices=[list(np.where(np.isnan(certain[i]) == False)[0]),
                                                              indices[i][0], indices[i][1]]))

        return

    # append from a list of uframe_instance objects
    def append_from_uframe_instance(self, instances, colnames=None):

        if len(self.columns) > 0:

            dimensions = len(
                [len(instance) for instance in instances if len(instance) == len(self.columns)])
            if dimensions != len(instances):
                raise ValueError("Dimensions of new instances do not match dimension of uframe")
            categoricals = [i for i in range(len(self.columns)) if self._col_dtype[i] == 'categorical']
            for index in categoricals:
                # check if that index is continuous or a certain float value in any of the new uframe_instances
                for instance in instances:
                    if index in instance.indices[1]:
                        raise ValueError("Continuous distribution in categorical column")
                    if index in instance.indices[0]:
                        certain_value = instance.certain_data[instance.indices[0].index(index)]
                        if certain_value - math.floor(certain_value) > 0:
                            raise ValueError("Float value in categorical column")

        # treat colnames parameter here in else case
        else:
            if colnames is None:
                self._columns = [*list(range(len(instances[0])))]
                self._colnames = {i: i for i in range(len(instances[0]))}
            else:
                if len(colnames) != len(instances[0]):
                    raise ValueError("Length of column list does not match size of value array")
                else:
                    self._columns = colnames
                    self._colnames = {name: i for i,
                                      name in enumerate(colnames)}
            self._col_dtype = len(self.columns) * ['continuous']

        # print("Data", self.data)
        # print("Instances", instances)
        self.data = self.data + instances
        # print("After appending", self.data)

        return

    # samples: list of length (n_instances) of np.ndarray of shape (dim_samples, n_samples) oder listen von np.arrays (dim_samples,)
    # erstellt kernels und greift auf bereits existierende append-Funktion zurück
    # für scipy Gaussian und alle sklearn Kernels
    def append_from_samples(self, samples_list, kernel='stats.gaussian_kde', colnames=None, rownames=None):

        if len(samples_list) < 1:
            raise ValueError("No samples given")

        kernel_list = []

        for i, samples in enumerate(samples_list):

            if isinstance(samples, list):
                samples = np.array(samples).T

            if kernel == 'stats.gaussian_kde':
                kde = stats.gaussian_kde(samples)
            elif kernel in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
                samples = samples.T
                kde = KernelDensity(kernel=kernel, bandwidth=1.0).fit(samples)
            else:
                raise NotImplementedError("Given kernel does not exist")
            kernel_list.append(kde)

        if kernel == 'stats.gaussian_kde':
            self.append_from_scipy_kde(kernel_list, colnames=colnames)
        else:
            self.append_from_sklearn_kde(kernel_list, colnames=colnames)

        self.addRownames(kernel_list, rownames)
        return

    def addRownames(self, new, rownames):

        if rownames is not None:

            # check rownames for duplicates (also with previously existing rownames)
            if len(set(self._rows + rownames)) != len(self.rows) + len(rownames):
                raise ValueError("Duplicates among rownames")
            if len(rownames) != len(new):
                raise ValueError(
                    "Number of rownames given does not match number of rows given")

            self._rows = self._rows + rownames
            self._rownames.update(
                {rownames[i]: len(self.data) - len(new) + i for i in range(len(rownames))})
        else:

            self._rows = self.rows + [*list(range(len(self.data) - len(new), len(self.data)))]
            # print("Rows", self.rows, "updated with", [*list(range(len(self.data) - len(new), len(self.data)))])
            self._rownames.update({i: i for i in range(
                len(self.data) - len(new), len(self.data))})

        return

    def __len__(self):
        return len(self.data)

    def __repr__(self):

        return "Object of class uframe"

    def __str__(self):

        print("Object of class uframe with certain values:")
        print(self.array_rep())
        return ""


    def indices_uncertain(self): 
        indices = [np.array(inst.indices[0]) for inst in self.data]
        uncertain = np.ones(self.shape)
        for i in range(len(indices)):
            if len(indices[i])>0: uncertain[i, indices[i]] = 0
        
        return uncertain.astype(np.bool)
        
        
        
    def array_rep(self):

        x = np.zeros((len(self.data), len(self.columns)), dtype=np.float64)
        for i, instance in enumerate(self.data):
            x[i, instance.indices[0]] = instance.certain_data
            x[i, instance.indices[1]] = np.nan
            x[i, instance.indices[2]] = np.nan

        return x

    @property
    def shape(self): 
        return self.sample().shape

    @shape.setter 
    def shape(self): 
        print("Can't manually set shape")
        
        
    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, new_columns):

        if len(self.columns) > 0 and len(new_columns) != len(self._columns):
            raise ValueError(
                "Length of new column list does not match column number of uframe")

        if len(set(new_columns)) != len(new_columns):
            raise ValueError("New column list contains duplicates!")

        # update _colnames dictionary if it is not empty

        new_d = {}
        for i, new_name in enumerate(new_columns):

            new_d[new_name] = self._colnames.pop(self.columns[i])

        self._colnames = new_d
        self._columns = new_columns
        return

    @property
    def rows(self):
        return self._rows

    # rows always have to be integers so far, probably keep it that way like pandas
    @rows.setter
    def rows(self, new_rows):

        if len(self.rows) > 0 and len(new_rows) != len(self._rows):
            raise ValueError(
                "Length of new row list does not match row number of uframe")

        if len(set(new_rows)) != len(new_rows):
            raise ValueError("New row list contains duplicates")

        # check that all row indices are integers
        if len([index for index in new_rows if isinstance(index, int)]) != len(new_rows):
            raise ValueError("Indices of rows have to be integers")

        new_d = {}
        for i, new_index in enumerate(new_rows):

            new_d[new_index] = self._rownames.pop(self.rows[i])

        self._rownames = new_d
        self._rows = new_rows

        return

    def get_row(self, key):
        return self._rownames[key]

    @property
    def col_dtype(self):

        # könnte das hier ändern, sodass es die Reihenfolge gemäß self.columns zurückgibt
        # noch nicht relevant, aber falls Ändern der Spaltenreihenfolge später möglich ist
        # nutze dafür dann self._colnames für Zuordnungen
        return self._col_dtype

    # set type of a column, if new type matches the values and distributions in that column
    # not tested yet
    def set_col_dtype(self, column, col_dtype):

        if col_dtype not in ['continuous', 'categorical']:
            raise ValueError("Invalid col_dtype")

        if column not in self._colnames.keys():
            raise ValueError("Column does not exist")

        col_index = self._colnames[column]
        if self._col_dtype[col_index] == col_dtype:
            #print("Column already has desired col_dtype")
            return
        if col_dtype == 'continuous':
            for instance in self.data:
                if col_index in instance.indices[2]:
                    raise ValueError("Categorical distributions in this column, cannot make it continuous")
        elif col_dtype == 'categorical':
            for instance in self.data:
                if col_index in instance.indices[1]:
                    raise ValueError("Continuous distributions in this column, cannot make it categorical")
                elif col_index in instance.indices[0]:
                    certain_value = instance.certain_data[instance.indices[0].index(col_index)]
                    if certain_value - math.floor(certain_value) > 0:
                        raise ValueError("Float value in column, cannot make it categorical")
        self._col_dtype[col_index] = col_dtype

        return

    # function for reordering columns and rows
    # Parameter: Liste mit Teilmenge der Spalten in einer bestimmten Reihenfolge, interpretiere das als
    # neue Reihenfolge dieser Spalten, i.e., wenn bei Spalten [1,2,4] der Parameter [4,2,1] übergeben wird,
    # werden diese 3 Spalten so neu angeordnet, der Rest bleibt unverändert
    # dies wird nur für die Ausgabe in columns gespeichert (passe dann array_rep Funktion entsprechend an)
    # Frage: wird dann auf Input für append Funktionen entsprechend eingegangen und gemäß
    # _colnames vertauscht vor dem Speichern? Müsste aus Gründen der Benutzerfreundlichkeit eigentlich so passieren
    # würde in den append-Funktionen einem geeigneten Umsortieren gemäß self._colnames entsprechen
    # noch zu besprechen
    def reorder_columns(self):

        raise NotImplementedError()

    def mode(self, **kwargs):

        return np.concatenate([inst.mode(**kwargs) for inst in self.data], axis = 0)

    def var(self, n:int = 50, seed: Optional[int] = None):
 
        return np.vstack([inst.var(n = n, seed = seed) for inst in self.data])

    def sample(self, n:int =1, seed: Optional[int] = None, threshold = 1):

        return np.concatenate([inst.sample(n = n, seed = seed, threshold = threshold) for inst in self.data], axis=0)

    def ev(self, n: Optional[int] = None, seed: Optional[int] = None):
        
        
        if n is None: n = 50 * self.shape[1]
        
        evs = [inst.ev(n,seed) for inst in self.data]
        if 'categorical' not in self._col_dtype:
            
            #print([(ev, type(ev)) for ev in evs])
            return np.concatenate([ev.reshape((1, len(ev))) for ev in evs], axis=0)
        else:
            #print("Evs", evs)
            
            #conts = [ev[0].shape for ev in evs]
            cats = []
            conts = []
            for i, ev in enumerate(evs):
                if type(ev) is list:
                    cats.append(ev[1])
                    conts.append(ev[0])
                else:
                    cats.append([])
                    conts.append(ev)
            
           # print("Length of cats", len(cats), cats)
            
            #print("Ev 0 list", [ev[0] for ev in evs])
            #print([ev[0].shape for ev in evs])
            #print("Conts", conts)
            #print([cont.shape for cont in conts])
            conts = np.concatenate([cont.reshape((1, len(cont))) for cont in conts], axis=0)
            #print("Conts", conts, conts.shape)
            ohes = {}
            for i in range(len(self.columns)):
                if self._col_dtype[i]=='categorical':
                
                    #have some known values in conts as well as nan values
                    
                    #have to iterate over lines
                    #know all levels of this variable from cat value dict
                    #need to make sure it exists 
                    levels = list(self.cat_value_dicts[i].keys())
                    m = np.zeros((conts.shape[0], len(levels)))
                    for j in range(conts.shape[0]):
                        #print("i", i, "j", j, "Indices", self.data[j].indices)
                        if np.isnan(conts[j,i]):
                            index = self.data[j].indices[2].index(i)
                            #print("I", i, "j", j, "Index", index)
                            for key in cats[j][index].keys():
                                #print("Cat value dict for column i", self.cat_value_dicts[i])
                                #print("Key", key)
                                m[j,int(self.cat_value_dicts[i][key])]= cats[j][index][key]
                        else:
                            m[j, int(conts[j,i])]=1
                    
                    ohes[i]= m
                    #print("Shape of ohes i", ohes[i].shape)
                
                #get new array from conts and the m arrays in ohes 
            ev_array = None
            for i in range(len(self.columns)):
                #print("i", i)
                if ev_array is None:
                    if self._col_dtype[i]=='categorical':
                        ev_array = ohes[i]
                    else:
                        ev_array = conts[:,i].reshape((conts.shape[0],1))
                else:
                    if self._col_dtype[i]=='categorical':
                        ev_array = np.concatenate((ev_array, ohes[i]), axis=1)
                    else:
                        ev_array = np.concatenate((ev_array, conts[:,i].reshape((conts.shape[0],1))), axis=1)
            
            return ev_array 
                
                    
    # performs One Hot Encoding of categorical columns after filling uncertain values by sampling/ mode/ ev
    # return columns in some way?
    def get_dummies(self, method='sample', samples_per_distrib=1):

        # do sampling, ev, modal
        # then iterate over the columns and for categorical
        if method == 'sample':
            x = self.sample(samples_per_distrib, None)
        elif method == 'mode':
            x = self.mode()
        elif method == 'ev':
            #expected value already incorporates ohe in categorical case 
            x = self.ev()
            return x 
        else:
            raise ValueError("Need method for treating uncertain values")
        
        
        return self.get_dummies_for_array(x)
        
    #gets dummies for given array x, which has to match the column number, no extra method there
    def get_dummies_for_array(self, x):
        
        x_ohe = None
        for i in range(x.shape[1]):
            if self._col_dtype[i] == 'categorical':

                integer_encoded = x[:,i]
                
                if hasattr(self, 'cat_values') and i in self.cat_values.keys():
                    categories = self.cat_values[i]
                else:
                    categories = []
                    
                for j in range(len(self.data)):
                    if i in self.data[j].indices[0]:
                        categories.append(self.data[j].certain_data[self.data[j].indices[0].index(i)])
                    elif i in self.data[j].indices[2]:

                        categories = categories + [float(key) for key in 
                                                   self.data[j].categorical[self.data[j].indices[2].index(i)].keys()]
                 
                categories = [np.array(list(set(categories)))]

                #print("Categories", categories)
                onehot_encoder = OneHotEncoder(sparse_output=False, categories= categories)
                integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                
                # print("One hot encoded shape", onehot_encoded.shape)
                if x_ohe is None:
                    x_ohe = onehot_encoded
                else:
                    x_ohe = np.concatenate((x_ohe, onehot_encoded), axis=1)

            else:
                if x_ohe is None:
                    x_ohe = x[:, i].reshape((len(x), 1))
                else:
                    x_ohe = np.concatenate((x_ohe, x[:, i].reshape((len(x), 1))), axis=1)

        return x_ohe
           
    def make_cat_value_dicts(self):
        
        self.cat_value_dicts = {}
        
        for i in range(len(self.columns)):
            
            #print("I", i)
            
            if self._col_dtype[i]=='categorical':
            
            #check if self has a cat values attribute
                if hasattr(self, 'cat_values') and i in self.cat_values.keys():
                    categories = self.cat_values[i]
                else:
                    categories = []
                
                for j in range(len(self.data)):
                    if i in self.data[j].indices[0]:
                        categories.append(self.data[j].certain_data[self.data[j].indices[0].index(i)])
                    elif i in self.data[j].indices[2]:
    
                        categories = categories + [float(key) for key in 
                                               self.data[j].categorical[self.data[j].indices[2].index(i)].keys()]
             
                categories = list(set(categories))
                #print("Categories", categories)
                self.cat_value_dicts[i]= {categories[i]:i for i in range(len(categories))}
            
            else:
                self.cat_value_dicts[i]= {}
        
        return 
    
    #new stuff as parameter: do not think so, would have to find out what new uframe instances are there
    def update_cat_value_dicts(self):
        
        #check for each categorical column if all its values are already in the cat value dict 
        #if not, append new keys with them and give them the correct index 
        
        
        return 
            
    
    def __getitem__(self, index):
                
        if type(index) == int: 
            index = [index]
        if type(index) == slice: 
            ret = uframe(new = self.data[index] , colnames = self._columns, rownames = self._rows[index])
            index = range(0, len(self))[index]
        else: 
            ret = uframe(new = [self.data[i] for i in index], colnames = self._columns, rownames = [self._rows[i] for i in index])

    
        for i,j  in enumerate(index): 
            if self.data[j]._mode_calculated():
                ret.data[i]._set_mode(self.data[j].mode())

        return ret                
        
   
    def add_cat_values(self, cat_value_dict):
        
        if hasattr(self, 'cat_values'):
            self.cat_values.update(cat_value_dict)
        else:
            self.cat_values = cat_value_dict
        
        self.update_cat_value_dicts()
        
        return 
    
    def ML(self, points, y): 
        
        if self.shape[1] != points.shape[1]: 
            raise ValueError(f"Wrong Dimension of given points")
        if self.shape[0] != y.shape[0]:
            raise ValueError(f"Wrong count of y values")
            
        pdfs = self.pdf(points)

        a = [pdfs[y==y_value].sum(axis=0) for y_value in np.unique(y)]
        
        dic = {}
        for y_value in np.unique(y): 
            dic[y_value] = pdfs[y==y_value].sum(axis=0)
        
        
        s = np.zeros(points.shape[0])
        for key in dic.keys():
            s += dic[key]
            
        for key in dic.keys():
            dic[key] = dic[key]/s
            
        return dic



    def pdf(self, points): 
    
        if type(points) == list: 
            points = np.array(points)
        
        ret = [self.data[i].pdf(points)for i in range(len(self.data))]

        ret = np.stack([arr.squeeze() for arr in ret])       
        
        return ret
     
    def save(self, name="saved_uframe.pkl"):
            
        l=[]
        for i in range(len(self)): 
            l.append([None if len(self.data[i].indices[0]) == 0 else self.data[i].certain_data, 
                      None if len(self.data[i].indices[1]) == 0 else self.data[i].continuous,
                      None if len(self.data[i].indices[2]) == 0 else self.data[i].categorical, 
                      self.data[i].indices])
            
        with open(name, 'wb') as f:
            pickle.dump(l,f)
        return 
     


    
   
   