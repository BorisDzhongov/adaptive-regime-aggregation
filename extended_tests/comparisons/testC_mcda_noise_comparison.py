import numpy as np
import csv

PHI = (1 + 5 ** 0.5) / 2
L = 5.0
PROJECTS = ["A", "B", "C", "D"]

G = np.array([
    [8.5,7.0,9.0,6.5],
    [7.5,8.5,6.5,6.2],
    [6.0,8.1,5.5,8.0],
    [6.5,7.0,5.0,8.5],
    [6.0,7.5,8.5,5.8],
    [7.0,6.0,8.0,6.5],
    [6.5,7.5,6.0,8.0],
    [5.5,6.5,4.5,7.5],
    [5.5,6.5,4.0,7.0],
    [6.0,6.5,5.0,7.5],
    [7.0,6.5,8.5,6.0]
])

I = np.array([
    [8.0,6.5,9.5,6.0],
    [7.0,9.0,6.0,7.0],
    [5.0,6.5,4.5,7.5],
    [6.0,7.5,4.0,8.0],
    [5.5,7.0,8.0,6.0],
    [6.5,5.0,7.0,6.0],
    [6.0,7.0,5.5,7.5],
    [4.0,5.5,3.0,7.0],
    [4.5,6.0,3.0,6.5],
    [5.0,6.0,4.0,7.0],
    [6.5,6.0,9.0,5.5]
])

M, N = G.shape


def clip(x):
    return np.clip(x,0,10)


def ara(g,i,alpha):
    return (L + g + alpha*i)/(2+alpha)


def mean_score(x):
    return x.mean(axis=0)


def topsis_score(x):

    denom = np.sqrt((x**2).sum(axis=1,keepdims=True))
    r = x/denom

    ideal_best = r.max(axis=1,keepdims=True)
    ideal_worst = r.min(axis=1,keepdims=True)

    d_pos = np.sqrt(((r-ideal_best)**2).sum(axis=0))
    d_neg = np.sqrt(((r-ideal_worst)**2).sum(axis=0))

    return d_neg/(d_pos+d_neg+1e-9)


def winners(scores):
    return np.argmax(scores,axis=0)


def run():

    n_mc = 50000
    rng = np.random.default_rng(42)

    g_true = np.repeat(G[:,:,None],n_mc,axis=2)
    i_true = np.repeat(I[:,:,None],n_mc,axis=2)

    eps = rng.normal(0,0.8,g_true.shape)

    g_obs = clip(g_true + eps)
    i_obs = clip(i_true + eps)

    methods = {
        "WSM":0.5*(g_obs+i_obs),
        "ARA_phi":ara(g_obs,i_obs,PHI),
        "ARA_phi2":ara(g_obs,i_obs,PHI**2)
    }

    rows=[]

    for name,x in methods.items():

        s_mean = mean_score(x)
        s_topsis = topsis_score(x)

        for rule,s in [("mean",s_mean),("topsis",s_topsis)]:

            w = winners(s)

            freq = np.bincount(w,minlength=N)/len(w)

            rows.append([
                name+"_"+rule,
                freq[0],freq[1],freq[2],freq[3]
            ])

    with open("testC_mcda_noise_results.csv","w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow(["method","P_A","P_B","P_C","P_D"])

        writer.writerows(rows)


if __name__=="__main__":
    run()
