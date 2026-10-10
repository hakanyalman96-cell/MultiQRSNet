"""
MultiQRSNet — regenerate Figure 5 (Grad-CAM, 4-panel) and Figure 6 (UNSUPERVISED
UMAP of held-out latent embeddings), addressing reviewer points on Figures 5 & 6.

Run on your Mac after the ablation script, in the same folder as the cache:

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    pip3 install torch numpy scikit-learn scipy matplotlib umap-learn --break-system-packages
    python3 multiqrsnet_figures.py

INPUT : multiqrsnet_cache_bin.npz  (X:(N,12,2500), y:(N,), grp:(N,))
OUTPUT: Figure5_GradCAM.png , Figure6_UMAP_unsupervised.png

Figure 6 is deliberately UNSUPERVISED: the UMAP is fit on out-of-fold GAP
embeddings WITHOUT using labels, so a visible separation is not circular.
Points are aggregated to one point per patient; misclassified patients are
marked with an x. Figure 5 shows Grad-CAM for four auto-selected cases
(confident-correct and borderline/incorrect, for each of RVOT and LVOT).
"""
import os, numpy as np, torch
from torch import nn, optim
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE="multiqrsnet_cache_bin.npz"; SEED=42; N_SPLITS=5; EPOCHS=50; BATCH=64
DEV=("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"))
print("device:",DEV)
LEADS=["V1","V2","V3","V4","V5","V6","I","II","III","aVL","aVR","aVF"]

# ---- preprocessing (identical to training) ----
def rn(x): m=np.median(x);q1,q3=np.percentile(x,[25,75]);return (x-m)/((q3-q1)+1e-6)
def mdi(l): s=l-np.median(l);mid=len(s)//2;w=s[mid-100:mid+100];return np.argmax(np.abs(w))/len(w)
def build(Xr):
    N=Xr.shape[0];X=np.zeros((N,24,2500),np.float32);M=np.zeros((N,12),np.float32)
    for i in range(N):
        for c in range(12):
            n=rn(Xr[i,c].astype(np.float32));X[i,c]=n;X[i,12+c]=np.gradient(n);M[i,c]=mdi(Xr[i,c])
    return X,M

# ---- model (matches multiqrsnet.py) ----
class SE(nn.Module):
    def __init__(s,ch,r=16):
        super().__init__();h=max(ch//r,4);s.fc=nn.Sequential(nn.Linear(ch,h),nn.ReLU(True),nn.Linear(h,ch),nn.Sigmoid())
    def forward(s,x): return x*s.fc(x.mean(2)).unsqueeze(2)
class RB(nn.Module):
    def __init__(s,ci,co,st=2,k=5):
        super().__init__();p=k//2
        s.c1=nn.Conv1d(ci,co,k,st,p);s.n1=nn.InstanceNorm1d(co,affine=True)
        s.c2=nn.Conv1d(co,co,k,1,p);s.n2=nn.InstanceNorm1d(co,affine=True);s.se=SE(co);s.d=None
        if st!=1 or ci!=co: s.d=nn.Sequential(nn.Conv1d(ci,co,1,st),nn.InstanceNorm1d(co,affine=True))
    def forward(s,x):
        idt=x if s.d is None else s.d(x);o=torch.relu(s.n1(s.c1(x)));o=s.n2(s.c2(o));o=s.se(o);return torch.relu(o+idt)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv1d(24,64,7,1,3),nn.InstanceNorm1d(64,affine=True),nn.ReLU(True),nn.MaxPool1d(2))
        s.rb1=RB(64,64);s.rb2=RB(64,128);s.rb3=RB(128,256)
        s.head=nn.Sequential(nn.Linear(268,128),nn.ReLU(True),nn.Dropout(0.5),nn.Linear(128,2))
        s.feat=None
    def forward(s,x,m):
        x=s.stem(x);x=s.rb1(x);x=s.rb2(x);x=s.rb3(x);s.feat=x   # (B,256,L) keep for Grad-CAM
        g=x.mean(2);return s.head(torch.cat([g,m],1))

def train(X,M,y,tr,ep=EPOCHS):
    net=Net().to(DEV);crit=nn.CrossEntropyLoss(label_smoothing=0.1)
    opt=optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-3)
    sch=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10)
    Xt=torch.tensor(X[tr]);Mt=torch.tensor(M[tr]);yt=torch.tensor(y[tr])
    for _ in range(ep):
        net.train();perm=torch.randperm(len(tr))
        for b in range(0,len(tr),BATCH):
            idx=perm[b:b+BATCH]
            xb=Xt[idx].to(DEV);mb=Mt[idx].to(DEV);yb=yt[idx].to(DEV)
            opt.zero_grad();crit(net(xb,mb),yb).backward();opt.step()
        sch.step()
    return net

def main():
    C=np.load(CACHE,allow_pickle=True);Xr=C["X"][:,:12,:].astype(np.float32);y=C["y"].astype(int);grp=C["grp"]
    torch.manual_seed(SEED);np.random.seed(SEED);X,M=build(Xr)
    sgkf=StratifiedGroupKFold(N_SPLITS,shuffle=True,random_state=SEED)
    emb=np.zeros((len(y),256),np.float32);prob=np.zeros(len(y),np.float32);last=None
    for tr,va in sgkf.split(X,y,grp):
        net=train(X,M,y,tr);net.eval();last=net
        with torch.no_grad():
            xb=torch.tensor(X[va]).to(DEV);mb=torch.tensor(M[va]).to(DEV)
            logit=net(xb,mb);emb[va]=net.feat.mean(2).cpu().numpy();prob[va]=torch.softmax(logit,1)[:,1].cpu().numpy()
    pred=(prob>0.5).astype(int)

    # ---------- Figure 6: UNSUPERVISED UMAP, patient-level ----------
    try:
        import umap
        reducer=umap.UMAP(n_neighbors=15,min_dist=0.1,random_state=SEED)  # unsupervised: no y
        Z=reducer.fit_transform(emb)
    except Exception as e:
        from sklearn.decomposition import PCA
        print("umap-learn not available, using PCA:",e);Z=PCA(2,random_state=SEED).fit_transform(emb)
    pats=sorted(set(grp))
    pZ=np.array([Z[grp==p].mean(0) for p in pats])
    pt=np.array([int(y[grp==p][0]) for p in pats])
    pcorr=np.array([ (pred[grp==p].mean()>0.5)==(y[grp==p][0]==1) for p in pats])
    plt.figure(figsize=(7,6))
    for cls,col,nm in [(1,"#c0392b","RVOT"),(0,"#2471a3","LVOT")]:
        m=(pt==cls)&pcorr;  plt.scatter(pZ[m,0],pZ[m,1],c=col,s=45,label=f"{nm} (correct)",edgecolors="none",alpha=.85)
        m=(pt==cls)&(~pcorr);plt.scatter(pZ[m,0],pZ[m,1],c=col,s=70,marker="x",linewidths=2,label=f"{nm} (misclassified)")
    plt.xlabel("UMAP-1 (arbitrary units)");plt.ylabel("UMAP-2 (arbitrary units)")
    plt.title("Unsupervised UMAP of held-out latent embeddings (per patient)")
    plt.legend(frameon=False,fontsize=9);plt.tight_layout();plt.savefig("Figure6_UMAP_unsupervised.png",dpi=300)
    print("saved Figure6_UMAP_unsupervised.png")

    # ---------- Figure 5: Grad-CAM, 4 panels ----------
    def gradcam(i):
        net=last;net.eval()
        xb=torch.tensor(X[i:i+1],requires_grad=True).to(DEV);mb=torch.tensor(M[i:i+1]).to(DEV)
        logit=net(xb,mb);cls=int(logit.argmax(1))
        net.zero_grad();logit[0,cls].backward()
        A=net.feat.detach()[0];  # (256,L)
        w=A.mean(1);cam=torch.relu((w[:,None]*A).sum(0)).cpu().numpy()  # (L,)
        cam=np.interp(np.linspace(0,len(cam)-1,2500),np.arange(len(cam)),cam)
        cam=(cam-cam.min())/(np.ptp(cam)+1e-9);return cam,cls
    # auto-select 4 cases
    def pick(cls,correct):
        mask=(y==cls)&((pred==y)==correct)
        if mask.sum()==0: return None
        ids=np.where(mask)[0]
        # most confident for correct, closest to 0.5 for borderline
        key=np.abs(prob[ids]-(1 if cls==1 else 0)) if correct else np.abs(prob[ids]-0.5)
        return ids[np.argmin(key)]
    cases=[("Confident correct LVOT",pick(0,True)),("Borderline/incorrect LVOT",pick(0,False)),
           ("Confident correct RVOT",pick(1,True)),("Borderline/incorrect RVOT",pick(1,False))]
    fig,axes=plt.subplots(4,1,figsize=(11,13))
    for ax,(title,i) in zip(axes,cases):
        if i is None: ax.set_title(title+" — none available");continue
        cam,cls=gradcam(i)
        sig=X[i,:12]  # normalized leads
        off=0
        for c in range(12):
            s=sig[c]/ (np.abs(sig[c]).max()+1e-6)*0.4
            ax.plot(np.arange(2500),s+off,color="k",lw=.5)
            ax.imshow(cam[None,:],aspect="auto",cmap="jet",alpha=.30,extent=[0,2500,off-0.5,off+0.5],vmin=0,vmax=1)
            ax.text(-90,off,LEADS[c],va="center",ha="right",fontsize=7);off-=1
        pr=prob[i]; ax.set_title(f"{title} | true={'RVOT' if y[i]==1 else 'LVOT'}  pred P(RVOT)={pr:.2f}  [ablation site: VERIFY {grp[i]}]",fontsize=10)
        ax.set_xlim(-140,2500);ax.set_yticks([]);ax.set_xlabel("samples (1 kHz)")
    plt.tight_layout();plt.savefig("Figure5_GradCAM.png",dpi=250);print("saved Figure5_GradCAM.png")

if __name__=="__main__": main()
