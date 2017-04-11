# ##### BEGIN GPL LICENSE BLOCK #####
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, version 2 of the license.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "UV SA Packing",
    "category": "UV",
    "description": "Pack UV islands to save space.",
    "author": "Tommi Hyppänen (ambi)",
    "location": "UV/Image Editor > Tools > Misc > UV SA Packing > Start pack",
    "version": (0, 0, 7),
    "blender": (2, 78, 0)
}

import bpy
from mathutils import geometry as gm
import math
import random
import time
import copy
from collections import namedtuple, OrderedDict, defaultdict
import cProfile, pstats, io
import bisect
import numpy as np
import sys
import heapq
from itertools import chain
import multiprocessing as mp

rnd = random.random
wm = bpy.context.window_manager
bpy_intersect = gm.intersect_line_line_2d

def ccw(a, b, c):
   return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


class Island():
    def __init__(self, polys, verts, margin):
        self.margin = margin
        
        self.rotation = 0   
        self.flipped = False
        self.scale = 1.0
        
        # verts matching poly indices
        self.verts = verts
        self.polys = polys

    def clone(self):
        cc = copy.copy
        i = Island(self.polys, self.verts, self.margin)
        i = cc(self)
        i.lines = np.copy(self.lines)
        i._bb_calc()
        return i

    def copy(self, o):
        cc = copy.copy
        self = cc(o)
        self.lines = np.copy(o.lines)
        self._bb_calc()

    def _bb_calc(self):
        x, y = self.lines[0][0]
        self.minx, self.miny, self.maxx, self.maxy = (x, y, x, y)
        
        for a in self.lines:
            for t in a:
                if t[0] > self.maxx: self.maxx = t[0]  
                if t[0] < self.minx: self.minx = t[0] 
                if t[1] > self.maxy: self.maxy = t[1]  
                if t[1] < self.miny: self.miny = t[1]    
            
        self.bb_x = self.maxx - self.minx
        self.bb_y = self.maxy - self.miny
        
        self.bb_size = round(self.bb_x * self.bb_y, 6)  


    def _updateloc(self, x, y):
        self.maxx += x
        self.minx += x
        self.maxy += y
        self.miny += y
        self.avg_x += x
        self.avg_y += y
 
        self.lines[:,:,0] += x
        self.lines[:,:,1] += y


    def nuke_inner_lines(self):
        """ Initialized and formats the island for collision testing """ 

        def _clockwise(edges):
            s = 0
            for e in edges:
                s += (e[1][0]-e[0][0])*(e[1][1]+e[0][1])
            return s > 0

        ## find rim edges
        ecount = 0
        edges = defaultdict(int)
        dir = defaultdict(bool)

        iloop = None
        cwise = None
        for i in range(len(self.verts)): 
            vertloop = tuple(tuple(j) for j in self.verts[i])
            prev_iloop = iloop
            iloop = [(vertloop[j], vertloop[(j+1) % len(vertloop)]) for j in range(0, len(vertloop), 1)]
            
            # Sort of a hack for detecting and fixing degenerate faces 
            # (for example :overlapping, because of cube uv projection)
            if cwise != None:
                t = _clockwise(iloop)
                if t != cwise:
                    break
            else:
                cwise = _clockwise(iloop)    
                
            # reverse loop direction
            if cwise:
                iloop.reverse()
                iloop = [(j[1], j[0]) for j in iloop]
            
            # for every edge in face...
            for (a, b) in iloop:
                ecount += 1
                d = a > b

                if a > b: # if a is greater, swap a and b
                    a, b = b, a
                edges[(a,b,)] += 1
                dir[(a,b,)] = d
            
        edges = {k: v for k, v in edges.items() if v == 1}

        ## generate margin
        # rebuild clockwise edges
        mlines = []
        for (a, b) in [k for k in edges.keys()]:
            if dir[(a, b)]:
                a, b = b, a
            mlines.append((a, b))

        chains = []
        
        chains.append([mlines[0]])
        mlines.remove(mlines[0])
        
        # connect edges to loop(s)
        def _findnext(ch, ln):
            c = ch[-1][-1]
            l = [a for a in ln if c[1] == a[0] and a not in ch[-1]]
            if l: 
                ln.remove(l[0])
                return l[0]    
            else:
                return None
        
        while mlines:
            i = _findnext(chains, mlines)
            if not i: 
                chains.append([mlines[0]]) 
                mlines.remove(mlines[0])
            else:
                chains[-1].append(i)

        for c in chains:
            assert len(c) > 1
            assert c[-1][1] == c[0][0]
            
        self.chains = chains

        # calculate normals
        # edge rot90        
        for ch in chains:
            lrot = []
            for (a, b) in ch:
                n = b[0] - a[0], b[1] - a[1]
                # 90 degree rotation
                n = np.array([-n[1], n[0]])
                n/= np.linalg.norm(n)
                lrot.append(n)
            # 2 edges norms normalized
            lnorms = []
            for i, l in enumerate(lrot):
                norm = lrot[i] + lrot[(i+1)%len(lrot)]
                norm = -(norm * self.margin)/np.linalg.norm(norm)
                lnorms.append(norm)

            assert len(lnorms) == len(ch)
            # move lines to create dilation      
            for j, val in enumerate(ch):
                x1, y1 = lnorms[j]
                x0, y0 = lnorms[(j-1)%len(lnorms)]
                ch[j] = ((ch[j][0][0] + x0, ch[j][0][1] + y0), (ch[j][1][0] + x1, ch[j][1][1] + y1))
                              
        # sort and remove dups
        self.lines = []
        for ch in chains:
            self.lines.extend(ch)
        self.lines = [(i[0],i[1]) if i[0]<i[1] else (i[1],i[0]) for i in self.lines]
        self.lines = list(set(self.lines)) 
        self.lines.sort()

        self.np_lines = np.empty(shape=(len(self.lines), 2, 2), dtype=np.float)
        for i, l in enumerate(self.lines):
            self.np_lines[i] = ((l[0][0], l[0][1]), (l[1][0], l[1][1]))
            
        self.lines = self.np_lines

        # bounding box
        self._bb_calc()

        self.avg_x = (self.maxx + self.minx)/2
        self.avg_y = (self.maxy + self.miny)/2

    def copy_position(self, other):
        self.move_vert(other.avg_x - self.avg_x, other.avg_y - self.avg_y)
        for i in range(other.rotation//90):
            self.rotate90_vert()
        if other.flipped:
            self.flipx_vert()
        self.scale_inplace(other.scale)
       
    def check_collision(self, other):        
        #### check bounding boxes
        if self.minx > other.maxx or self.miny > other.maxy or self.maxx < other.minx or self.maxy < other.miny:
            return False        

        #### check for point inside polygon collision, leftmost point
        def _onepoint():
            x, y = self.lines[0][0]
            if x > other.lines[0][0][0]:
                inside = False
                iline = ((-1.0, y), self.lines[0][0])
                for ol in other.lines:
                    if ol[0][0] > x:
                        break
                    #if bpy_intersect(*iline, *ol): 
                    if (ol[0][1]<y) != (ol[1][1]<y) and (ol[0][0]+(ol[1][0]-ol[0][0])*(y-ol[0][1])/(ol[1][1]-ol[0][1]) < x):
                        inside = not inside

                if inside:
                    return True
            return False
        
        if _onepoint():
            return True
                    
        #### check for line collisions 
        test_lines = []
        oidx = 0

        # find start of self lines in other lines
        while oidx < len(other.lines) and other.lines[oidx][1][0] < self.lines[0][0][0]:
            oidx += 1

        def _linecoll(oidx):
            for i, l in enumerate(self.lines):
                # add more other lines to the window
                while oidx < len(other.lines) and other.lines[oidx][0][0] <= l[1][0]:
                    heapq.heappush(test_lines, (other.lines[oidx][1][0], other.lines[oidx].tolist()))
                    oidx += 1
                
                # remove lines from the window that are outside
                while test_lines and test_lines[0][1][1][0] < l[0][0]:
                    heapq.heappop(test_lines)
                
                for ol in test_lines:
                    if bpy_intersect(l[0], l[1], ol[1][0], ol[1][1]):
                        return True
                                    
            return False

        return _linecoll(oidx)

    
    def check_collision_verify(self, other):
        # check bounding boxes
        if self.minx > other.maxx or self.miny > other.maxy or self.maxx < other.minx or self.maxy < other.miny:
            return False        
        
        # check for point inside polygon collision, random
        a = self.verts[random.randint(0, len(self.verts)-1)]
        for t in a:
            for f in other.verts: # faces
                if len(f) == 4:
                    if gm.intersect_point_quad_2d(t, f[0], f[1], f[2], f[3]):
                        return True
                elif len(f) == 3:
                    if gm.intersect_point_tri_2d(t, f[0], f[1], f[2]):
                        return True
                    
        # check for line collisions
        for l in self.lines:
            for ol in other.lines:
                if gm.intersect_line_line_2d(l[0], l[1], ol[0], ol[1]):
                    return True

        return False 

        
    def _rebuild_lines(self):
        self.lines = np.array([(i if i[0][0]<i[1][0] else [i[1], i[0]]) for i in self.lines])
        self.lines = self.lines[self.lines[:,0,0].argsort()]

    def rotate90(self):
        x = np.copy(self.lines[:,:,0])
        y = self.lines[:,:,1]
        
        self.lines[:,:,0] = self.avg_x + self.avg_y - y
        self.lines[:,:,1] = self.avg_y + x - self.avg_x 
                                     
        self._rebuild_lines()
        self._bb_calc()
        self.rotation = (self.rotation + 90) % 360
        
    def flipx(self):
        axis = (self.rotation//90)%2
        avg = self.avg_x if axis == 0 else self.avg_y
        self.lines[:,:,axis] = avg - (self.lines[:,:,axis] - avg)
                    
        self._rebuild_lines()
        self._bb_calc()
        self.flipped = not self.flipped
        
    def rotate90_vert(self):
        def _tpoint(x, y):
            return self.avg_x + self.avg_y - y, self.avg_y + x - self.avg_x
            
        for i in range(len(self.verts)):
            for j in range(len(self.verts[i])):
                x, y = self.verts[i][j]
                self.verts[i][j] = list(_tpoint(x, y))
        
        self._bb_calc()
        self.rotation = (self.rotation + 90) % 360
        
    def flipx_vert(self):
        if self.rotation == 0 or self.rotation == 180:
            for i in range(len(self.verts)):
                for j in range(len(self.verts[i])):
                    x = self.verts[i][j][0]
                    self.verts[i][j][0] = self.avg_x-(x-self.avg_x)
                    
        elif self.rotation == 90 or self.rotation == 270:
            for i in range(len(self.verts)):
                for j in range(len(self.verts[i])):
                    y = self.verts[i][j][1]
                    self.verts[i][j][1] = self.avg_y-(y-self.avg_y)
                    
        self._bb_calc()
        self.flipped = not self.flipped
        
    def scale_from(self, scale):
        def _tpoint(x, y):
            if self.x_fill_dir == 1:
                x *= scale
            else:
                x = 1.0-(1.0-x)*scale
            
            if self.y_fill_dir == 1:
                y *= scale
            else:
                y = 1.0-(1.0-y)*scale
            return x, y
        
        for i in range(len(self.verts)):
            for j in range(len(self.verts[i])):
                self.verts[i][j][0], self.verts[i][j][1] = _tpoint(self.verts[i][j][0], self.verts[i][j][1])
                                  
        for i, (a, b) in enumerate(self.lines):
            self.lines[i] = (_tpoint(a[0], a[1]), _tpoint(b[0], b[1]))       
            
        self.avg_x, self.avg_y = _tpoint(self.avg_x, self.avg_y)
            
        self.scale *= scale
        self._bb_calc()       
        
    def scale_inplace(self, scale):
        def _tpoint(x, y):
            x = self.avg_x + (x - self.avg_x) * scale    
            y = self.avg_y + (y - self.avg_y) * scale    
            return x, y
        
        for i in range(len(self.verts)):
            for j in range(len(self.verts[i])):
                self.verts[i][j][0], self.verts[i][j][1] = _tpoint(self.verts[i][j][0], self.verts[i][j][1])
                
        for i, (a, b) in enumerate(self.lines):
            self.lines[i] = (_tpoint(a[0], a[1]), _tpoint(b[0], b[1]))    
                           
        self.scale *= scale    
        self._bb_calc()    
        
    def move(self, x, y):
        self._updateloc(x, y)
        
    def move_vert(self, x, y):
        for i in range(len(self.verts)):
            for j in range(len(self.verts[i])):
                self.verts[i][j][0] += x
                self.verts[i][j][1] += y
        self._updateloc(x, y)
        
    def set_location(self, x, y):
        dispx = x - self.minx
        dispy = y - self.miny 
        self._updateloc(dispx, dispy)        


class UV():
    def __init__(self, obj):
        self.obj = obj
        self.uv_layer = obj.data.uv_layers.active.data
        self.uv_esp = 5
        self.islands = [] 
        
        print("Finding islands...")
        
        # all polys
        selfpolys = self.obj.data.polygons
        
        print("..polys")
        uvlr = self.uv_layer
        print(len(selfpolys))
        
        _loc2poly = defaultdict(set)
        _poly2loc = defaultdict(set)
        for i, poly in enumerate(selfpolys):
            uvs = set((round(uvlr[i].uv[0], self.uv_esp),
                       round(uvlr[i].uv[1], self.uv_esp)) for i in poly.loop_indices)
            for u in uvs:
                _loc2poly[u].add(poly.index)
                _poly2loc[poly.index].add(u)    

        def _parse(poly):
            out = set()
            if poly in _poly2loc: 
                out.add(poly)
                connected = _poly2loc[poly]
                del _poly2loc[poly]
                
                # while connections found
                while connected:
                    new_connections = set()
                    for c in connected:
                        for p in _loc2poly[c]:
                            out.add(p)
                            new_connections |= _poly2loc[p]
                            del _poly2loc[p]
                    connected = new_connections

            return out
        
        # while unprocessed vertices remain... 
        isles = []       
        while _poly2loc:
            # get random poly from existing points
            _poly = next(iter(_poly2loc.items()))[0]
            
            # repeat until no more connected vertices (remove connected vertices)                              
            isles.append(_parse(_poly))

        print("..done")
        self.islefaces = []
        for i in isles:
            self.islefaces.append(list(i))        

        print(len(self.islefaces))
            
    def write(self, island):
        for fi, f in enumerate(island.polys):      
            for i, loop_idx in enumerate(self.obj.data.polygons[f].loop_indices):
                self.obj.data.uv_layers.active.data[loop_idx].uv = island.verts[fi][i]

    def generate_isle_set(self, margin):
        isles = []
        for i in range(len(self.islefaces)):
            verts = []
            polys = self.islefaces[i]
            uv_layer = self.obj.data.uv_layers.active
            for f in polys:       
                lidcs = self.obj.data.polygons[f].loop_indices
                tar = []
                for l in lidcs:
                    x, y = round(uv_layer.data[l].uv[0], 6), round(uv_layer.data[l].uv[1], 6)
                    tar.append([x, y])
                verts.append(tar)

            isle = Island(polys, verts, margin)
            isle.nuke_inner_lines()
            isles.append(isle)

        isles.sort(key=lambda x: 1-x.bb_size)
        return isles    
        
    def get_uv_area(self):
        def triangle_area(verts):
            # Heron's formula
            a = (verts[1][0]-verts[0][0])**2.0 + (verts[1][1]-verts[0][1])**2.0 
            b = (verts[2][0]-verts[1][0])**2.0 + (verts[2][1]-verts[1][1])**2.0 
            c = (verts[0][0]-verts[2][0])**2.0 + (verts[0][1]-verts[2][1])**2.0
            cal = (2*a*b + 2*b*c + 2*c*a - a**2 - b**2 - c**2)/16
            if cal<0: cal=0 
            return math.sqrt(cal)

        def quad_area(verts):
            return triangle_area(verts[:3]) + triangle_area(verts[2:]+[verts[0]])
        
        total_area = 0.0
        for poly in self.obj.data.polygons:
            if len(poly.loop_indices) == 3:
                total_area += triangle_area([self.uv_layer[i].uv for i in poly.loop_indices])
            if len(poly.loop_indices) == 4:
                total_area += quad_area([self.uv_layer[i].uv for i in poly.loop_indices])

        return total_area

class ShotgunPackingOperator(bpy.types.Operator):
    bl_idname = "uv.shotgunpack"
    bl_label = "UV Simulated Annealing Packer"

    _updating = False
    _calcs_done = False
    _timer = None
    _p = -1
    
    def init_run(self):
        bpy.ops.object.mode_set(mode='OBJECT')

        print("Starting packing...")
        self.uv_hnd = UV(bpy.context.object)
        
        print("Generate manipulation set.")
        self.isles = self.uv_hnd.generate_isle_set(bpy.context.scene.s_packing_margin)
        print("Generate reference set.")
        self.orig_isles = self.uv_hnd.generate_isle_set(bpy.context.scene.s_packing_margin)
        
        self.uv_area = self.uv_hnd.get_uv_area()
        print("UV area:", self.uv_area)
        print("Scaling...")
        percentg = bpy.context.scene.s_packing_percentage

        for isle in self.isles:
            isle.scale_inplace(percentg/self.uv_area)    

        # start monte carlo iteration
        self.iters = bpy.context.scene.s_packing_iterations
        self.start_time = time.time()
        self.total_checks = 0

        self.current_scale = 1.0
    
    def run(self, p):
        # find a free location
        thisisle = self.isles[p]
        thisisle.scale_inplace(self.current_scale)
        
        allisle = self.isles        
        NITERS = self.iters
        
        somecollision = True
        while somecollision:
            xdir, ydir = 1, 1
            thisisle.x_fill_dir = xdir
            thisisle.y_fill_dir = ydir 
                        
            def _findplace(isle, iters):
                collides = False
                
                # go through search space
                for xi in range(iters):                          
                    if rnd() > 0.2:
                        isle.rotate90()
                    else:
                        isle.flipx()
                    
                    rnd_x = (1.0 - isle.bb_x)/float(iters-1)
                    rnd_y = (1.0 - isle.bb_y)/float(iters-1)
                    
                    new_x = float(xi) * rnd_x
                    isle.set_location(new_x, 0)
                    
                    in_range = [i for i in allisle[:p] if not (i.minx>isle.maxx or i.maxx<isle.minx)]
                    for yi in range(iters):          
                        new_y = float(yi) * rnd_y

                        isle.set_location(new_x, new_y)
                        
                        collides = False
                        for itf in in_range:
                            if isle.check_collision(itf):
                                collides = True   
                          
                        if not collides:     
                            break
                    if not collides:
                        break
                return collides
                    
            somecollision = _findplace(thisisle, NITERS)
                
            if somecollision:
                # scale back
                print("couldn't find free room. rescaling 0.99x"," scale:",round(self.current_scale,3))
                self.current_scale *= 0.99
                thisisle.scale_inplace(0.99)
                for tf in range(0,p):
                    allisle[tf].scale_from(0.99)
                
        # finished packing current island
        print(repr(p)+"/"+repr(len(self.isles)))
        self.total_checks += 0
            
    def end_run(self):    
        end_time = time.time()
        tot_time = end_time - self.start_time
        print(repr(self.total_checks) + " total collision checks in " + repr(round(tot_time,2))+ \
            " seconds. (" + repr(round(float(self.total_checks)/tot_time, 2)) + " checks per second)")
        print("End of packing.")

        # write data out
        for i, isle in enumerate(self.orig_isles):
            isle.copy_position(self.isles[i])
            self.uv_hnd.write(isle)

    def execute(self, context):
        self.init_run()
        for p in range(len(self.isles)):
            self.run(p)
        self.end_run()
        return {'FINISHED'}

    
class UVSPackerPanel(bpy.types.Panel):
    """UV Packing Panel"""
    bl_label = "UV SA Packing"
    bl_idname = "uv.sgpackpanel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'TOOLS'

    def draw(self, context):
        layout = self.layout

        #if context.scene.uv_packing_progress:
        #    row = layout.row()
        #    row.label(text="Progress: "+context.scene.uv_packing_progress)
        
        row = layout.row()
        row.prop(context.scene, "s_packing_iterations", text="Island iterations")

        row = layout.row()
        row.prop(context.scene, "s_packing_percentage", text="Start percentage")

        row = layout.row()
        row.prop(context.scene, "s_packing_margin", text="Margin")

        row = layout.row()
        row.operator(ShotgunPackingOperator.bl_idname, text="Start pack")

def register():
    bpy.utils.register_class(ShotgunPackingOperator)
    bpy.utils.register_class(UVSPackerPanel)

    bpy.types.Scene.uv_packing_progress = bpy.props.StringProperty(
        name="uv_packing_progress", description="UV Packing Progress")
        
    bpy.types.Scene.s_packing_iterations = bpy.props.IntProperty(name="s_packing_iterations", default=50, min=1, max=1000)
    bpy.types.Scene.s_packing_percentage = bpy.props.FloatProperty(name="s_packing_percentage", default=0.7, max=1.0, min=0.1)
    bpy.types.Scene.s_packing_margin = bpy.props.FloatProperty(name="s_packing_margin", default=0.01, max=0.5, min=0.0)
    

def profile_debug():
    pr = cProfile.Profile()
    pr.enable()
    bpy.ops.uv.shotgunpack()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats(sortby).print_stats()
    print(s.getvalue())

def unregister():
    bpy.utils.unregister_class(ShotgunPackingOperator)
    bpy.utils.unregister_class(UVSPackerPanel)

if __name__ == "__main__":
    print("executing.")
    #unregister()
    register()
    profile_debug()
    print("finished.")

   
    
