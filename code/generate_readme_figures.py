"""Rebuild README figures from the project's saved results, without retraining.

Run from the repository root: python code/generate_readme_figures.py
Requires numpy and Pillow. Original counts are preserved below so the
figures can be rebuilt offline. Sources:
- code/code.ipynb: saved severity counts, classification reports, and summary.
- figures/confusion_matrix_logistic_basline.jpg
- figures/confusion_matrix_distilbert_baseline.jpg
- figures/confusion_matrix_domain_adapt.jpg
The confusion-matrix counts were transcribed from those existing figures and
cross-checked against the notebook's support, accuracy, macro-F1, and cost risk.
All evaluation figures describe the same fixed 2,237-example test set and
one saved run per model; they do not imply repeated-run uncertainty estimates.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
CLASSES = ['Routine', 'Elevated', 'Urgent', 'Critical']
MODELS = ['TF-IDF + Logistic Regression', 'DistilBERT', 'Domain-adapted DistilBERT']
COLORS = ['#536675', '#0072B2', '#D55E00']
HATCHES = ['///', '', '...']
SPLIT_COUNTS = np.array([
    [8554, 4362, 424, 268],
    [1494, 626, 71, 46],
    [1427, 678, 86, 46],
])
CMS = np.array([
    [[1018,302,85,22], [217,380,56,25], [27,36,16,7], [6,6,4,30]],
    [[1149,259,11,8], [267,384,9,18], [41,31,9,5], [7,6,2,31]],
    [[1117,281,21,8], [265,387,12,14], [41,30,10,5], [9,17,1,19]],
])
COST = np.array([[0,1,3,4], [1,0,2,4], [3,2,0,2.5], [5,4,2.5,0]])
EXPECTED = [
    # accuracy, macro-F1, average risk, total risk: notebook cell outputs
    [.6455,.4722,.5849,1308.5],
    [.7032,.5221,.4213,942.5],
    [.6853,.4760,.4613,1032.0],
]

def validate():
    assert (SPLIT_COUNTS.sum(axis=1) == [13608,2237,2237]).all()
    expected_recall = np.array([[.713,.560,.186,.652],[.805,.566,.105,.674],[.783,.571,.116,.413]])
    for k, (cm, expected) in enumerate(zip(CMS, EXPECTED)):
        assert np.array_equal(cm.sum(axis=1), SPLIT_COUNTS[2])
        diag = np.diag(cm)
        precision = diag / cm.sum(axis=0)
        recall = diag / cm.sum(axis=1)
        f1 = 2*precision*recall/(precision+recall)
        risk = (cm*COST).sum()
        observed = [diag.sum()/cm.sum(), f1.mean(), risk/cm.sum(), risk]
        assert np.allclose(observed, expected, rtol=0, atol=.000051), (MODELS[k],observed)
        assert np.allclose(recall, expected_recall[k], rtol=0, atol=.000501)
    return CMS.diagonal(axis1=1, axis2=2)/CMS.sum(axis=2)

# Pillow renders directly to PNG so this helper needs no plotting backend.
from PIL import Image, ImageDraw, ImageFont
from html import escape
import math

SCALE=2
FONT_CANDIDATES=[
    Path('C:/Windows/Fonts/arial.ttf'),
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    Path('/Library/Fonts/Arial.ttf'),
]
FONT_PATH=next((p for p in FONT_CANDIDATES if p.exists()),None)
if FONT_PATH is None:
    raise RuntimeError('Install Arial or DejaVu Sans to render readable figures.')
INK='#26313C'; MUTED='#536270'; GRID='#DFE5EA'
SVG=[]

def font(size): return ImageFont.truetype(str(FONT_PATH),round(size*SCALE))

def canvas(w,h):
    SVG.clear()
    SVG.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    SVG.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    image=Image.new('RGB',(w*SCALE,h*SCALE),'white')
    return image,ImageDraw.Draw(image)

def text(draw,xy,label,size=22,fill=INK,anchor='la'):
    draw.text(tuple(int(v*SCALE) for v in xy),str(label),font=font(size),fill=fill,anchor=anchor)
    align={'l':'start','m':'middle','r':'end'}[anchor[0]]
    x,y=xy
    y=y+size*.91 if anchor[1]=='a' else y+size*.33 if anchor[1]=='m' else y
    SVG.append(f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{align}" font-family="Arial, DejaVu Sans, sans-serif" font-size="{size}" fill="{fill}">{escape(str(label))}</text>')

def line(draw,xy,fill=GRID,width=1):
    draw.line([round(v*SCALE) for v in xy],fill=fill,width=round(width*SCALE))
    x1,y1,x2,y2=xy
    SVG.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{fill}" stroke-width="{width}"/>')

def rect(draw,xy,fill):
    draw.rectangle([round(v*SCALE) for v in xy],fill=fill)
    x1,y1,x2,y2=xy
    color=fill if isinstance(fill,str) else '#%02x%02x%02x'%fill
    SVG.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{x2-x1:.2f}" height="{y2-y1:.2f}" fill="{color}"/>')

def save(image,name):
    image.save(OUT/name,optimize=True,dpi=(180,180))
    (OUT/Path(name).with_suffix('.svg')).write_text('\n'.join(SVG+['</svg>'])+'\n',encoding='utf-8')

def pattern(draw, box, kind):
    x0,y0,x1,y1=box
    if kind==0:
        for yy in range(int(y0)+8,int(y1)-3,14):
            line(draw,(x0+2,yy,x1-2,min(yy+9,y1-1)),fill='#B9C2C9',width=1)
    elif kind==2:
        for yy in range(int(y0)+7,int(y1)-2,14):
            for xx in range(int(x0)+7,int(x1)-2,14):
                rect(draw,(xx,yy,xx+1,yy+1),'#FFD3BD')

def grouped(values, names, ylabel, title, filename, supports=None, counts=None, note=''):
    image,draw=canvas(1320,670)
    text(draw,(660,32),title,31,anchor='ma')
    # Same legend order and bar order support comparison without color alone.
    legend_x=[130,520,810]
    for k,(name,col,xx) in enumerate(zip(names,COLORS,legend_x)):
        rect(draw,(xx,96,xx+29,117),col); pattern(draw,(xx,96,xx+29,117),k)
        text(draw,(xx+40,106),name,20,anchor='lm')
    left,right,top,bottom=112,1275,165,528
    vmax=100 if supports is not None else 80
    for v in range(0,vmax+1,20):
        yy=bottom-v/vmax*(bottom-top)
        line(draw,(left,yy,right,yy))
        text(draw,(left-15,yy),f'{v}%',21,MUTED,'rm')
    line(draw,(left,top,left,bottom),fill=MUTED)
    line(draw,(left,bottom,right,bottom),fill=MUTED)
    text(draw,(left,top-30),ylabel,19,MUTED)
    group=(right-left)/4; bw=60
    for j,label in enumerate(CLASSES):
        center=left+(j+.5)*group
        text(draw,(center,552),label,25,anchor='ma')
        if supports is not None: text(draw,(center,588),f'n = {supports[j]:,}',20,MUTED,'ma')
        for k in range(3):
            value=values[k,j]; xx=center+(k-1)*70
            yy=bottom-value/vmax*(bottom-top)
            rect(draw,(xx-bw/2,yy,xx+bw/2,bottom-1),COLORS[k])
            pattern(draw,(xx-bw/2,yy,xx+bw/2,bottom-1),k)
            text(draw,(xx,yy-9),f'{value:.1f}%',20,anchor='ms')
            if counts is not None:
                text(draw,(xx,yy-36),f'{counts[k,j]:,}',18,MUTED,'ms')
    text(draw,(660,644),note,19,MUTED,'ma')
    save(image,filename)

def normalized():
    image,draw=canvas(1050,1930)
    text(draw,(525,27),'Row-normalized test confusion matrices',31,anchor='ma')
    text(draw,(525,74),'Each row totals 100%; original counts appear in parentheses.',21,MUTED,'ma')
    x0,cell=220,150
    for k,(name,cm) in enumerate(zip(MODELS,CMS)):
        top=180+k*560
        text(draw,(520,top-59),name,27,anchor='ma')
        vals=cm/cm.sum(axis=1,keepdims=True)*100
        for i,label in enumerate(CLASSES):
            yy=top+i*100
            text(draw,(x0-18,yy+36),label,23,anchor='rm')
            text(draw,(x0-18,yy+65),f'n = {cm[i].sum():,}',18,MUTED,'rm')
            for j in range(4):
                xx=x0+j*cell
                t=vals[i,j]/100
                low=np.array([242,247,252]); high=np.array([8,48,107])
                rgb=tuple(np.round(low+(high-low)*t).astype(int))
                rect(draw,(xx,yy,xx+cell-1,yy+99),rgb)
                fill='white' if vals[i,j]>=55 else INK
                text(draw,(xx+cell/2,yy+37),f'{vals[i,j]:.1f}%',27,fill,'mm')
                text(draw,(xx+cell/2,yy+70),f'({cm[i,j]:,})',21,fill,'mm')
        for j,label in enumerate(CLASSES):
            text(draw,(x0+(j+.5)*cell,top+417),label,23,anchor='ma')
        text(draw,(520,top+455),'Predicted severity',21,MUTED,'ma')
    # One fixed, shared percentage scale for all panels.
    lx,ly,lh=900,450,800
    for step in range(lh):
        t=1-step/(lh-1); low=np.array([242,247,252]); high=np.array([8,48,107])
        rect(draw,(lx,ly+step,lx+28,ly+step+1),tuple(np.round(low+(high-low)*t).astype(int)))
    for pct in [0,25,50,75,100]:
        yy=ly+(1-pct/100)*lh
        text(draw,(lx+40,yy),f'{pct}%',21,MUTED,'lm')
    text(draw,(525,1895),'Fixed test set: n = 2,237. Rows represent true severity; columns represent predictions.',19,MUTED,'ma')
    save(image,'confusion_matrices_normalized.png')

def main():
    recall=validate()
    grouped(SPLIT_COUNTS/SPLIT_COUNTS.sum(axis=1,keepdims=True)*100,
            ['Train (n = 13,608)','Validation (n = 2,237)','Test (n = 2,237)'],
            'Share within split','Severity distribution after mapping',
            'severity_distribution_by_split.png',counts=SPLIT_COUNTS,
            note='Original splits before oversampling. Bar labels show percentages; gray labels show counts.')
    grouped(recall*100,['TF-IDF + Logistic Reg.', 'DistilBERT', 'Domain-adapted DistilBERT'],'Recall','Test recall by severity class',
            'per_class_recall_comparison.png',supports=SPLIT_COUNTS[2],
            note='Same fixed test set (n = 2,237); one saved run per model. n denotes true-class support.')
    normalized()
    print('Verified all source supports and published metrics; wrote 3 figures in PNG and SVG formats.')

if __name__=='__main__': main()
