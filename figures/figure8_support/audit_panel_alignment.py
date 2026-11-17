"""Release-local geometry check for Figure 8's four-panel layout.

This small adapter is newly authored for the release. It does not redistribute
the general-purpose external figure-auditing skill used in the original work.
It changes no data or artists and fails on misaligned panel rectangles.
"""
import json
from pathlib import Path


def require_matplotlib_panel_alignment(fig, *, json_out, overlay_svg=None,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True):
    fig.canvas.draw()
    width,height=fig.get_size_inches()*72
    panels=[]
    for ax in fig.axes:
        spec=ax.get_subplotspec()
        box=ax.get_position()
        panels.append(dict(row=spec.rowspan.start,column=spec.colspan.start,
            left=box.x0*width,right=box.x1*width,bottom=box.y0*height,top=box.y1*height))
    errors=[]
    if len(panels)!=4 or {(p['row'],p['column']) for p in panels}!={(0,0),(0,1),(1,0),(1,1)}:
        errors.append('Expected exactly the Figure 8 two-by-two panel grid')
    else:
        by={(p['row'],p['column']):p for p in panels}
        for row in (0,1):
            for edge in ('bottom','top'):
                if abs(by[row,0][edge]-by[row,1][edge])>tolerance_pt:errors.append(f'Row {row} {edge} alignment')
        for col in (0,1):
            for edge in ('left','right'):
                if abs(by[0,col][edge]-by[1,col][edge])>tolerance_pt:errors.append(f'Column {col} {edge} alignment')
        for edge1,edge2 in (('left','right'),('bottom','top')):
            spans=[p[edge2]-p[edge1] for p in panels]
            if max(spans)-min(spans)>tolerance_pt:errors.append('Unequal comparable panel span')
        h=[by[r,1]['left']-by[r,0]['right'] for r in (0,1)]
        v=[by[0,c]['bottom']-by[1,c]['top'] for c in (0,1)]
        if abs(h[0]-h[1])>gutter_tolerance_pt or abs(v[0]-v[1])>gutter_tolerance_pt:
            errors.append('Unequal gutters')
    if require_panel_labels:
        labels={t.get_text().strip().strip('()').lower() for ax in fig.axes for t in ax.texts}
        labels|={t.get_text().strip().strip('()').lower() for t in fig.texts}
        if not {'a','b','c','d'}<=labels:errors.append('Missing explicit panel labels')
    report=dict(adapter='release_local_2x2_geometry_v1',passed=not errors,errors=errors,
                tolerance_pt=tolerance_pt,gutter_tolerance_pt=gutter_tolerance_pt,panels=panels)
    Path(json_out).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if overlay_svg:
        rects=''.join(f'<rect x="{p["left"]}" y="{height-p["top"]}" width="{p["right"]-p["left"]}" height="{p["top"]-p["bottom"]}" fill="none" stroke="red"/>' for p in panels)
        Path(overlay_svg).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">{rects}</svg>')
    if strict and errors:raise ValueError('Panel alignment failed: '+'; '.join(errors))
    return report
