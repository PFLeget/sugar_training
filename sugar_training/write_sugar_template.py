import pickle
import os
import numpy as np
import sugar_training as st

def write_sugar(path_output = 'data_output/', model = 'sugar_model_2.58_Rv.pkl'):

    sed = os.path.join(path_output, model)
    dic = pickle.load(open(sed))
    Rv = dic['Rv']

    fichier = open(os.path.join(path_output, 'SUGAR_model_v1.asci'),'w') 

    Time = np.linspace(-12,48,21)

    for Bin in range(len(dic['m0'])):
        fichier.write('%.5f    %.5f    %.5f    %.5f    %.5f    %.5f    %.5f    %.5f \n'%((Time[Bin%21],
                                                                                          dic['X'][Bin],
                                                                                          dic['m0'][Bin],
                                                                                          dic['alpha'][Bin,0],
                                                                                          dic['alpha'][Bin,1],
                                                                                          dic['alpha'][Bin,2],
                                                                                          st.extinctionLaw(dic['X'][Bin],Rv=Rv),1)))


    fichier.close()

    from operator import itemgetter

    infile = open(os.path.join(path_output, 'SUGAR_model_v1.asci'), 'r')
    lines = infile.readlines()
    infile.close()

    listik = []
    for line in lines:
        new_line = [float(i) for i in line.split()]
        listik.append(new_line)

    s = sorted(listik, key=itemgetter(0))

    names = ['0','1','2','3','4']
    for i in names:
        outfile = open(os.path.join(path_output, 'sugar_template_' + i + '.dat'), 'w')
        for line in s:
            j = 2+int(i)
            outfile.write('%4.4f %8.8f %8.8f' %(line[0],line[1],line[j]))
            outfile.write('\n')
        outfile.close()

    os.system('mkdir ' + os.path.join(path_output,'sugar_template_v1'))
    os.system('mv ' + os.path.join(path_output, 'sugar_template_* ') +  os.path.join(path_output,'sugar_template_v1'))
