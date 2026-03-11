import numpy as np
import csv

PHI=(1+5**0.5)/2
L=5

PROJECTS=["A","B","C","D"]

G=np.array([
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

I=np.array([
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

M,N=G.shape


def clip(x):
    return np.clip(x,0,10)


def ara(g,i,alpha):
    return (L+g+alpha*i)/(2+alpha)


def mean_score(x):
    return x.mean(axis=0)


def winners(scores):
    return np.argmax(scores,axis=0)


def run():

    n_mc=50000
    rng=np.random.default_rng(123)

    g_true=np.repeat(G[:,:,None],n_mc,axis=2)
    i_true=np.repeat(I[:,:,None],n_mc,axis=2)

    g_obs=g_true.copy()
    i_obs=i_true.copy()

    outliers=rng.random(g_obs.shape)<0.05

    g_obs=clip(g_obs + outliers*rng.normal(0,2,g_obs.shape))
    i_obs=clip(i_obs + outliers*rng.normal(0,2,i_obs.shape))

    missing=rng.random(g_obs.shape)<0.1

    g_obs[missing]=L
    i_obs[missing]=L

    methods={
        "WSM":0.5*(g_obs+i_obs),
        "ARA_phi":ara(g_obs,i_obs,PHI),
        "ARA_phi2":ara(g_obs,i_obs,PHI**2)
    }

    rows=[]

    for name,x in methods.items():

        scores=mean_score(x)

        w=winners(scores)

        freq=np.bincount(w,minlength=N)/len(w)

        rows.append([
            name,
            freq[0],freq[1],freq[2],freq[3]
        ])

    with open("testD_missing_outlier_results.csv","w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow(["method","P_A","P_B","P_C","P_D"])

        writer.writerows(rows)


if __name__=="__main__":
    run()
