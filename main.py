import numpy as np
import networkx as nx
from sklearn import linear_model
from sklearn.cluster import k_means
from sklearn.metrics import normalized_mutual_info_score
import math
from tqdm import tqdm
import multiprocessing
import numba
import matplotlib.pyplot as plt
import time


dataset_path=[
    # 'Amazon/amazon.ungraph.txt',
    # 'DBLP/dblp.ungraph.txt',
    # 'Livejournal/lj.ungraph.txt',
    # 'Orkut/orkut.ungraph.txt',
    # 'Youtube/youtube.ungraph.txt',

    # 'line.gml',
    # 'karate.gml',
    # 'football.gml',
    'polblogs.gml',
]
SIGMA_S_DATA={
    'line.gml':1,
    'karate.gml':1,
    'football.gml':10,
    'polblogs.gml':5,
}
KKK_DATA={
    'line.gml':2,
    'karate.gml':3,
    'football.gml':12,
    'polblogs.gml':7,
}


def input_G(path):
    f = open('./dataset/'+path,'r')
    f.readlines(4)
    g = {}
    for line in f:
        if line[0] == '#':
            continue
        uv=line.split('\t')
        u,v = uv[0],uv[1]
        if u in g:
            g[u].append(v)
        else:
            g[u]=[v]
        if v in g:
            g[v].append(u)
        else:
            g[v]=[u]
    G = {}
    for g_key in g:
        G[g_key] = np.asarray(g[g_key])
    # print(len(G))
    return G


def input_G_gml(path):
    Graph = nx.read_gml('./dataset/'+path,label='id')
    g = {}
    for (u,v,duplicated) in Graph.edges:
        if duplicated == 1:
            continue
        if u in g:
            g[u].append(v)
        else:
            g[u]=[v]
        # if not Graph.is_directed():
        if v in g:
            g[v].append(u)
        else:
            g[v]=[u]
        if v not in g:
            g[v]=[]
    G = {}
    for g_key in g:
        G[g_key] = np.asarray(g[g_key])
    # print(len(G))
    print('Graph.is_directed',Graph.is_directed())
    print('len(Graph.edges)',len(Graph.edges))
    return G


Id2Index={}
Index2Id={}
INF_DIS=1e8
def get_A_P(G):
    P = {}
    Id2Index.clear()
    Index2Id.clear()
    for s in G:
        if s not in Id2Index:
            Id2Index[s] = len(Id2Index)
            Index2Id[len(Index2Id)] = s
        P[s] = {}
        for g in G:
            P[s][g]=INF_DIS
        P[s][s] = 0
        dl,L,R = [s],0,1
        while L<R:
            now = dl[L]
            L+=1
            for v in G[now]:
                if P[s][v]==INF_DIS:
                    dl.append(v)
                    R+=1
                    P[s][v] = P[s][now]+1
                    # if len(P[s])%1000==0:
                    #     print('???',len(P[s]))
        # print('over',s,len(P[s]))
    # print(len(P))
    # print(P.keys())
    # print(list(P.keys()))
    # print(len(P.keys()),max(P.keys()))
    # print(len(Id2Index),Id2Index)
    # print(len(Index2Id),Index2Id)
    tmp_n = len(Index2Id)
    A=np.array([[1 if Index2Id[j] in G[Index2Id[i]] else 0 for j in range(tmp_n)] for i in range(tmp_n)])
    return A, P


def P2S(P, Sigma_s):
    S = []
    for i in range(len(P.keys())):
        idI = Index2Id[i]
        S.append([])
        for j in range(len(P.keys())):
            idJ = Index2Id[j]
            tmp_s = 1 if i==j else (0 if idJ not in P[idI] or P[idI][idJ]==INF_DIS else math.exp(-P[idI][idJ]*P[idI][idJ]/(Sigma_s*Sigma_s)))
            S[i].append(tmp_s)
    # print(np.array(S))
    return np.array(S)


def get_sparse_linear_decomposition(_lambda, column_id, S_Hat, Si):
    lasso = linear_model.Lasso(alpha=_lambda, copy_X=True, max_iter=10000,fit_intercept=False, precompute=True, warm_start=True, selection='random')
    lasso.fit(S_Hat, Si)
    if np.count_nonzero(lasso.coef_)<=3:
        if _lambda>=1e-5:
            return get_sparse_linear_decomposition(_lambda*0.66,column_id,S_Hat,Si)
        # print("\nZEROMMP\n\n\n\n")
        # print(Si)
        # print("maxSi:",np.max(Si))
        # print("\n\n\n\n\n")
        return np.array([0 if i != column_id else 1 for i in range(len(Si))])
    # print("Lasso Coef_","alpha:", _lambda," Non zero:", np.count_nonzero(lasso.coef_),"\tscore:", lasso.score(S_Hat, Si))
    return lasso.coef_ / sum(lasso.coef_)


def get_symmetric_linear_coefficient(S, alpha):
    n = S.shape[0]
    f = np.zeros(S.shape)
    zero_column = np.zeros([n, 1])
    for i in tqdm(range(n)):
        # S_Hat = np.column_stack((S[:, :i], zero_column, S[:, i + 1:]))
        S_Hat = np.copy(S)
        S_Hat[:,i] = 0
        Si = S[:, i]
        # print('\tSi.shape:',Si.shape)
        ai = get_sparse_linear_decomposition(alpha, i, S_Hat, Si)
        ai = ai / max(ai)
        f[i, :] = ai
    return f


def find_low_error_clusters(EVs,EVa,K,cluster_num):
    # print(EVs[0][0],EVs[1][0],EVs[2][0])
    # print(EVs[0][0],EVs[0][1],EVs[0][2])
    # Es = np.array([np.array(EVs).transpose()[i] for i in range(K)]).transpose()
    # print([len(np.array(EVs).transpose()[i]) for i in range(K)])
    # print('???',np.array(EVs).shape,np.array(EVs).transpose().shape)
    # EsAAAA = np.array([EVs[i] for i in range(K)])
    # print(Es.shape,Es[0],Es[3])
    # print(EsAAAA.shape,EsAAAA[0])
    # print(EsAAAA.transpose()[0])

    # Es = np.array([np.array(EVs).transpose()[i] for i in range(K)]).transpose()
    # Ea = np.array([np.array(EVa).transpose()[i] for i in range(K)]).transpose()
    Es=np.array([EVs[i] for i in range(K)]).transpose() #more natural
    Ea=np.array([EVa[i] for i in range(K)]).transpose()
    # print(Es.shape, Es[0][0], Es[1][0])
    E= np.column_stack((Ea[:, :], Es[:, :]))
    # print(E.shape, E[0][K], E[1][K])
    centroid, labels, inertia, best_n_iter = k_means(E.real,copy_x=True,n_init='auto', n_clusters=cluster_num, return_n_iter=True,random_state=233,max_iter=3000)
    # print('\tlabels:',labels)
    # print('inertia:', inertia)
    return labels,inertia


def get_line_k(X,Y,p):
    x1,y1,x2,y2=X[-p],Y[-p],X[-1],Y[-1]
    x1/=len(G)#len(X)
    x2/=len(G)#len(X)
    y1/=Y[0]
    y2/=Y[0]
    return -(y2-y1)/(x2-x1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.close()
    axis=None
    if len(dataset_path)>1:
        _,axis=plt.subplots(max(2,len(dataset_path)),max(2,len(dataset_path)))
    else:
        axis = plt

    np.set_printoptions(linewidth=150)
    for path in dataset_path:
        cntAxis = axis if len(dataset_path)==1 else axis[0][dataset_path.index(path)]
        # G = input_G(path)
        print('\n---------------------------------------------------------\n','PATH:',path)
        P_S_TIME = -time.time()
        G = input_G_gml(path)
        A, P = get_A_P(G)                                        #   Eq.1
        S = P2S(P,SIGMA_S_DATA[path])     #math.sqrt(1/math.log(1/0.85,math.e))
        P_S_TIME+=time.time()
        # print('A\n',A,'\n')
        # print('S\n',S,'\n')
        F_TIME = -time.time()
        F = get_symmetric_linear_coefficient(S, 0.1)
        # print('LASSO\n',F,'\n')
        F = (F + np.transpose(F)) / 2                       #   Eq.5
        F_TIME+=time.time()
        # print('F\n',F,'\n')
        # print('jjjjj',min([np.sum(F[:,i]) for i in range(F.shape[0])]))
        Ds = np.diag(np.array([np.sum(F[:,i]) for i in range(F.shape[0])]))
        # print('DS_diag\n',np.array([np.sum(F[:,i]) for i in range(F.shape[0])]),'\n')
        Da = np.diag(np.array([np.sum(A[:,i]) for i in range(A.shape[0])]))
        Ds12 = np.diag(np.array([math.sqrt(1.0/np.sum(F[:,i])) for i in range(F.shape[0])]))
        # print('DS12_diag\n',np.array([math.sqrt(1.0/np.sum(F[:,i])) for i in range(F.shape[0])]),'\n')
        Da12 = np.diag(np.array([math.sqrt(1.0/max(1,np.sum(A[:,i]))) for i in range(A.shape[0])]))
        # print('DEBUG np.sum(F[:,1])',np.sum(F[:,1]),'Ds[1]',Ds[1],'Ds12[1]',Ds12[1],'np.sum(Ds12.dot(F).dot(Ds12)[:,1])',np.sum(Ds12.dot(F).dot(Ds12)[:,1]),'np.sum(Ds12.dot(F)[:,1])',np.sum(Ds12.dot(F)[1,:]))
        # print('MMP Ds12.dot(F)',Ds12.dot(F))
        # print('MMP F.dot(Ds12)', F.dot(Ds12))
        # print("check SUMF",np.sum(F),np.sum(Ds12.dot(F)),np.sum(Ds12.dot(F).dot(Ds12)),F.shape)
        # print(Ds)
        # print(Da)
        # print('Ds12.dot(F).dot(Ds12)',Ds12.dot(F).dot(Ds12))
        Ls = np.identity(Ds.shape[0])-Ds12.dot(F).dot(Ds12)
        LsAAA = Ds-F
        La = np.identity(Da.shape[0])-Da12.dot(A).dot(Da12)
        # print('sumF',[np.sum(F[:,i]) for i in range(F.shape[0])])
        # print('sumLs',[np.sum(Ls[i]) +np.sum(Ls[:,i]) for i in range(Ls.shape[0])])
        # print('sumLsAAA',[np.sum(LsAAA[i]) for i in range(LsAAA.shape[0])])
        # print(Ls[1])
        # print('Ls\n',Ls,'\n')
        # print('La',La,'\n')
        (EVsVal,_EVs),(EVaVal,_EVa) = (np.linalg.eig(Ls),np.linalg.eig(La))
        # print('unorder Evs:',EVsVal)
        # print(EVs[1])
        EVs,EVa = [],[]
        for i in range(len(EVsVal)):
            EVs.append([_EVs[j][i] for j in range(len(EVsVal))])
            EVa.append([_EVa[j][i] for j in range(len(EVsVal))])
        EVs,EVa = zip(list(EVsVal),list(EVs)),zip(list(EVaVal),list(EVa))
        # print(list(EVs))
        EVs,EVa=sorted(EVs,key=lambda x:x[0]),sorted(EVa,key=lambda x:x[0])
        # print('EVs least K',[EVs[i][0] for i in range(KKK_DATA[path])],[EVa[i][0] for i in range(KKK_DATA[path])],'all EVs:',[x[0] for x in EVs])
        # print(EVs[0])
        EVs,EVa = [list(x[1]) for x in EVs],[list(x[1]) for x in EVa]       # least significant EV
        # print('Node33:',[x[Id2Index[33]] for x in EVs])
        # print('Node34:', [x[Id2Index[34]] for x in EVs])
        # print('test\n',Ls.dot(np.array(EVs[0]).transpose()),np.array(EVs[0]).transpose())
        # print('TZvec\n',EVs[0],'\n',Ls.dot(np.array(EVs[0]).transpose()))
        # print('TZvec\n',EVa[0],'\n',La.dot(np.array(EVa[0]).transpose()))
        # print(EVs[1])
        # print(EVs[0])
        # print(EVs[0:KKK_DATA[path]])
        # plt.plot(EVs[0],EVs[1])
        # find_low_error_clusters(EVs,EVa,K=3,cluster_num=4)
        # print(S[2])
        # print(S[3])
        Kerr=[]
        # print(len(EVs))
        fig_X=[]
        kmeans_threshold = math.e
        kmeans_cluster = 2
        KMEANS_TIME=-time.time()
        while kmeans_cluster<len(EVs)-3:
            fig_X.append(kmeans_cluster)
            cnt_labels,tmp=find_low_error_clusters(EVs, EVa, K=KKK_DATA[path], cluster_num=kmeans_cluster)
            # Kerr.append(math.sqrt(tmp))
            Kerr.append(math.sqrt(tmp)/kmeans_cluster)
            kmeans_cluster=max(kmeans_cluster+1,int(kmeans_cluster*1.1))
            if len(fig_X)>=5:
                minK=1e9
                for i in range(3,5,1):
                    minK=min(minK,get_line_k(fig_X,Kerr,i))
                # print('cluster:',kmeans_cluster,'Kerr',Kerr[-1],'lineK',minK)
                if minK<=kmeans_threshold:
                    break
        KMEANS_TIME += time.time()
        if len(dataset_path)==1:
            cntAxis.title(path[0:path.index('.')])
        else:
            cntAxis.set_title(path[0:path.index('.')])
        normKErr=[x/Kerr[0] for x in Kerr]
        cntAxis.plot(fig_X,normKErr,'ob-',markersize=3)
        fig_X.append(len(EVs))
        normKErr.append(0)
        cntAxis.plot(fig_X[-2:],normKErr[-2:],'ob:',markersize=3)

        labels, _ = find_low_error_clusters(EVs, EVa, K=KKK_DATA[path], cluster_num=fig_X[-2])
        communities={}
        for index,lb in enumerate(labels):
            if lb not in communities:
                communities[lb]=[index]
            else:
                communities[lb].append(index)
        for com in communities.values():
            avgD=0
            for i in com:
                for j in com:
                    if i!=j:
                        avgD+=min(P[Index2Id[j]][Index2Id[i]],P[Index2Id[i]][Index2Id[j]])
            if len(com)>1:
                avgD/=len(com)**2.0-len(com)
            print('|com|=',len(com),'avgDis:',avgD)
        print('number of communities:',len(communities))
        with open('./result/'+path[0:path.index('.')]+'_communities.txt', 'w') as f:
            for com in sorted(list(communities.values()),key=lambda x:-len(x)):
                f.write('\t'.join([str(Index2Id[i]) for i in com]))
                f.write('\n')
        with open('./result/'+path[0:path.index('.')]+'_time.txt', 'w') as f:
            f.write('P_S_time:'+str(P_S_TIME)+'\n')
            f.write('F_TIME:' + str(F_TIME) + '\n')
            f.write('KMEANS_TIME:' + str(KMEANS_TIME) + '\n')
    plt.show()

