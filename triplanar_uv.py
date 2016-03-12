# Calculate used UV area, works only for tri+quad meshes
# doesn't account for UV overlap

# Copyright 2016 Tommi Hyppänen
# License: GPL 2

import bpy
import math

ob = bpy.context.object
uv_layer = ob.data.uv_layers.active.data

def triangle_area(verts):
    # Heron's formula
    # uses optimization by Iñigo Quilez
    # http://www.iquilezles.org/blog/?p=1579
    # A² = (2ab + 2bc + 2ca – a² – b² – c²)/16
    a = (verts[1][0]-verts[0][0])**2.0 + (verts[1][1]-verts[0][1])**2.0 
    b = (verts[2][0]-verts[1][0])**2.0 + (verts[2][1]-verts[1][1])**2.0 
    c = (verts[0][0]-verts[2][0])**2.0 + (verts[0][1]-verts[2][1])**2.0
    cal = (2*a*b + 2*b*c + 2*c*a - a**2 - b**2 - c**2)/16
    if cal<0: cal=0 
    return math.sqrt(cal)

def quad_area(verts):
    return triangle_area(verts[:3]) + triangle_area(verts[2:]+[verts[0]])

def get_uv_area():
    total_area = 0.0
    for poly in ob.data.polygons:
        if len(poly.loop_indices) == 3:
            total_area += triangle_area([uv_layer[i].uv for i in poly.loop_indices])
        if len(poly.loop_indices) == 4:
            total_area += quad_area([uv_layer[i].uv for i in poly.loop_indices])

    return total_area

def approximate_islands():
    islands = []

    # merge polygons sharing uvs
    for poly in ob.data.polygons:
        uvs = set((round(uv_layer[i].uv[0], 2), round(uv_layer[i].uv[1], 2)) for i in poly.loop_indices)
        notfound = True
        for i,island in enumerate(islands):
            if island & uvs != set():
                islands[i] = islands[i].union(uvs)
                notfound = False
                break
        if notfound:
            islands.append(uvs)
        
    # merge sets sharing uvs
    for isleidx in range(len(islands)):    
        el = islands[isleidx]
        notfound = True
        for i, isle in enumerate(islands):
            if i == isleidx:
                continue
            if el & isle != set():
                islands[i] = islands[i].union(el)
                islands[isleidx] = set()
                notfound = False
                break
            
    # remove empty sets
    islands = [i for i in islands if i != set()]
    return islands
        

print(repr(len(approximate_islands())) + " approximate UV islands counted.")
print(repr(round((1.0-get_uv_area())*100, 2) )+"% UV area wasted.")
            