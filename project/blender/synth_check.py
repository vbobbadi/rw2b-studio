"""Draw saved labels over generator renders: green = visible joint, red = hidden by an occluder.
    python synth_check.py <generator_out_dir> <sheet.png>
"""
import numpy as np, sys
from PIL import Image, ImageDraw
d=sys.argv[1]; L=np.load(f"{d}/labels.npz"); n=len(L["focal"])
par=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
tiles=[]
for i in range(min(n, 12)):
    s=int(L["start"])+i
    row=[]
    for kind in ("clean","occ3d"):
        im=Image.open(f"{d}/{kind}_{s:06d}.png").convert("RGBA")
        bg=Image.new("RGBA",im.size,(40,40,40,255)); bg.alpha_composite(im); dr=ImageDraw.Draw(bg)
        j=L["joints2d"][i]; vis=L["vis_occ3d"][i]
        for c in range(1,24): dr.line([tuple(j[par[c]]),tuple(j[c])],fill=(255,255,0),width=1)
        for k,(x,y) in enumerate(j):
            col=(0,255,0) if (kind=="clean" or vis[k]) else (255,0,0)
            dr.ellipse([x-2,y-2,x+2,y+2],fill=col)
        if kind=="occ3d": dr.text((4,4),f"occ {L['occfrac_occ3d'][i]:.2f} n={L['n_occ'][i]}",fill=(255,255,255))
        row.append(bg)
    tiles+=row
W=256; cols=4; rows=(len(tiles)+cols-1)//cols
sheet=Image.new("RGB",(cols*W,rows*W))
for k,t in enumerate(tiles): sheet.paste(t.convert("RGB"),((k%cols)*W,(k//cols)*W))
sheet.save(sys.argv[2]); print("vis frac", L["vis_occ3d"].mean(), "occfrac", L["occfrac_occ3d"].round(2), "selfvis", L["selfvis"].mean(1).round(2))
