# -*- coding: utf-8 -*-

# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Curvature to vertex colors",
    "category": "Object",
    "description": "Set object vertex colors according to mesh curvature",
    "author": "Tommi Hyppänen (ambi)",
    "location": "3D View > Object menu > Curvature to vertex colors",
    "version": (0, 1, 7),
    "blender": (2, 79, 0)
}

import bpy
import random
from collections import defaultdict
import mathutils as mu
import math
import bmesh
import numpy as np
import cProfile, pstats, io


def read_verts(mesh):
    mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
    mesh.vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(mesh.vertices), 3))      


def read_edges(mesh):
    fastedges = np.zeros((len(mesh.edges)*2), dtype=np.int) # [0.0, 0.0] * len(mesh.edges)
    mesh.edges.foreach_get("vertices", fastedges)
    return np.reshape(fastedges, (len(mesh.edges), 2))


def read_norms(mesh):
    mverts_no = np.zeros((len(mesh.vertices)*3), dtype=np.float)
    mesh.vertices.foreach_get("normal", mverts_no)
    return np.reshape(mverts_no, (len(mesh.vertices), 3))


def read_polygon_verts(mesh):
    polys = np.zeros((len(mesh.polygons)*4), dtype=np.uint32)
    mesh.polygons.foreach_get("vertices", faces)
    return polys


def safe_bincount(data, weights, dts, conn):
    #for i, v in enumerate(edge_a):
    #    vecsums[v] += totdot[i]
    #    connections[v] += 1
    bc = np.bincount(data, weights)
    dts[:len(bc)] += bc
    bc = np.bincount(data)
    conn[:len(bc)] += bc
    return (dts, conn)


def tri_area(a, b, c):
    ab = (b - a)
    ac = (c - a)
    angle = ab.angle(ac)
    return ab.length * ac.length * math.sin(angle)/2.0

v = tri_area(mu.Vector((1,0,0)), mu.Vector((1,1,0)), mu.Vector((0,0,0)))
assert(v > 0.499 and v < 0.501)


"""
LAPLACIAN
for(int i : vertices)
{
  for(int j : one_ring(i))
  {
    for(int k : triangle_on_edge(i,j))
    {
      L(i,j) = cot(angle(i,j,k));
      L(i,i) -= cot(angle(i,j,k));
    }
  }
}

for(triangle t : triangles)
{
  for(edge i,j : t)
  {
    L(i,j) += cot(angle(i,j,k));
    L(j,i) += cot(angle(i,j,k));
    L(i,i) -= cot(angle(i,j,k));
    L(j,j) -= cot(angle(i,j,k));
  }
}
"""

def edgewise_gaussian(mesh):
    """ Gaussian from edges """
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    gaussian = []
    for i, vi in enumerate(bm.verts):
        # generate one ring
        ring_verts = [e.other_vert(vi) for e in vi.link_edges]

        area = 0
        for ri in range(len(ring_verts)):
            rv = ring_verts[ri]
            area += tri_area(rv.co, ring_verts[(ri+1) % len(ring_verts)].co, vi.co)
            #area += (rv.co - vi.co).length

        aG = 0.0
        mT = 0.0
        for j, vj in enumerate(ring_verts):
            m=0.0
            next_v = ring_verts[(j+1) % len(ring_verts)]
            prev_v = ring_verts[(j-1) % len(ring_verts)]
            t0, t1 = next_v.co - vi.co, vj.co - vi.co

            m = (tri_area(vi.co, vj.co, next_v.co) + tri_area(vi.co, vj.co, prev_v.co))/2/area
            #m = t1.length / area
            mT += m
            aG += t0.dot(t1)*m

        #print(mT)
        # mT == 1.0
        #K = (2*math.pi - aG)/area
        K = (2*math.pi - aG)/(math.pi*4)
        gaussian.append(K)

    return np.array(gaussian)


def laplace_beltrami(mesh):
    """ Laplace-Beltrami operator """
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    print("lp: begin")

    def _cot(a):
        return 1.0/math.tan(a) if a!=0 else 1.0

    gaussian = []
    k1 = []
    k2 = []
    rH = []

    bork = 0

        #print(len(ring_verts), len(vi.link_edges))

        # test validity
        # for rvi, rv in enumerate(ring_verts):
        #     p_v = ring_verts[(rvi-1) % len(ring_verts)]
        #     n_v = ring_verts[(rvi+1) % len(ring_verts)]

        #     print(rv, rvi)
        #     buddies = [e.other_vert(rv).index for e in rv.link_edges]

        #     if p_v.index not in buddies:
        #         print("P_V error!", buddies, "|", p_v.index, "| rv:",[e.index for e in ring_verts])

        #     if n_v.index not in buddies:
        #         print("N_V error!", buddies, "|", n_v.index, "| rv:",[e.index for e in ring_verts])

        #     print("---")

    for i, vi in enumerate(bm.verts):
        # generate one ring
        ring_verts = [e.other_vert(vi) for e in vi.link_edges]

        if len(ring_verts) < 3:
            print("1-ring length error at", i)

        assert(vi in [e.other_vert(ring_verts[0]) for e in ring_verts[0].link_edges])
        
        # one ring tri area 
        area = 0
        for ri in range(len(ring_verts)):
            rv = ring_verts[ri]
            area += tri_area(rv.co, ring_verts[(ri+1) % len(ring_verts)].co, vi.co)

        # temporary (_tri_area doesn't work right)
        #area = sum(f.calc_area() for f in vi.link_faces)
        #area /= 3.0

        #area = sum(e.calc_length() for e in vi.link_edges)

        aG = 0.0
        #H = mu.Vector((0, 0, 0))
        H = 0

        # plane equation for normal: n.dot(x-x0) = 0
        x0, y0, z0 = vi.co
        A, B, C = vi.normal
        D = -A*x0 -B*y0 -C*z0
        sqrABC = math.sqrt(A**2 + B**2 + C**2)    

        # roll right
        # ring_verts = [ring_verts[-1]] + ring_verts[:-1]

        msum = 0
        for j, vj in enumerate(ring_verts):
            prev_v = ring_verts[(j-1) % len(ring_verts)]
            next_v = ring_verts[(j+1) % len(ring_verts)]

            x1, y1, z1 = vj.co
            sn = (A*x1 + B*y1 + C*z1 + D)/sqrABC

            t0, t1, t2 = next_v.co - vi.co, vj.co - vi.co, prev_v.co - vi.co

            a = (t2).angle((prev_v.co - vj.co))
            b = (t0).angle((next_v.co - vj.co))

            m = _cot(a) + _cot(b)

            #m = (_tri_area(vi.co, vj.co, next_v.co) + _tri_area(vi.co, vj.co, prev_v.co))/2/area
            fj = t1.length
            H += m * t1.length
            #H += m * vj.co

            msum += m

            #aG += t0.dot(t1)
            aG += t0.dot(t1)/t1.length


        #K = (2*math.pi - aG)/(area/3)
        K = 2*math.pi - aG
        gaussian.append(K)

        #H -= vi.co
        #H = H.length
        H = H/(area/3)
        t = H**2 - K
        if t<0: 
            bork += 1
            t=0
        t = math.sqrt(t)

        k1.append(H + t)
        k2.append(H - t)

        rH.append(H)


    print("bork:",bork,"/",len(bm.verts))

    print("lp: end")

    return np.array(gaussian), np.array(k1), np.array(k2), np.array(rH)


def calc_normals(fastverts, fastnorms, fastedges):
    """ Calculates curvature for specified mesh """
    edge_a, edge_b = fastedges[:,0], fastedges[:,1]
    
    tvec = fastverts[edge_b] - fastverts[edge_a]
    tvlen = np.linalg.norm(tvec, axis=1)    

    # normalize vectors
    tvec = (tvec.T / tvlen).T 

    # adjust the minimum of what is processed   
    edgelength = tvlen * 100
    edgelength = np.where(edgelength < 1, 1.0, edgelength)

    vecsums = np.zeros(fastverts.shape[0], dtype=np.float) 
    connections = np.zeros(fastverts.shape[0], dtype=np.float) 

    # calculate normal differences to the edge vector in the first edge vertex
    totdot = (np.einsum('ij,ij->i', tvec, fastnorms[edge_a]))/edgelength
    safe_bincount(edge_a, totdot, vecsums, connections)

    # calculate normal differences to the edge vector  in the second edge vertex
    totdot = (np.einsum('ij,ij->i', -tvec, fastnorms[edge_b]))/edgelength
    safe_bincount(edge_b, totdot, vecsums, connections)

    # (approximate gaussian) curvature is the average difference of 
    # edge vectors to surface normals (from dot procuct cosine equation)
    #curve = 1.0 - np.arccos(vecsums/connections)/np.pi

    # 1 = max curvature, 0 = min curvature, 0.5 = zero curvature
    #curve -= 0.5
    #curve /= np.max([np.amax(curve), np.abs(np.amin(curve))])
    #curve += 0.5
    return np.arccos(vecsums/connections)/np.pi


def average_parameter_vector(data, fastverts, fastedges):
    """ Calculates normalized world-space vector that points into the direction of where parameter is higher on average """
    vecsums = np.zeros((fastverts.shape[0], 3), dtype=np.float) 
    connections = np.zeros(fastverts.shape[0], dtype=np.float) 

    for i, e in enumerate(fastedges):
        vert_a, vert_b = e
        vecsums[vert_a] += (data[vert_b] - data[vert_a]) * (fastverts[vert_b] - fastverts[vert_a])
        vecsums[vert_b] += (data[vert_a] - data[vert_b]) * (fastverts[vert_a] - fastverts[vert_b])
        connections[vert_a] += 1
        connections[vert_b] += 1

    divr = connections * np.linalg.norm(vecsums, axis=1) * data

    vecsums[:,0] /= divr
    vecsums[:,1] /= divr
    vecsums[:,2] /= divr

    return vecsums
            

def mesh_smooth_filter_variable(data, fastverts, fastedges, iterations):
    """ Smooths variables in data [-1, 1] over the mesh topology """
    # vert indices of edges
    edge_a, edge_b = fastedges[:,0], fastedges[:,1]
    tvlen = np.linalg.norm(fastverts[edge_b] - fastverts[edge_a], axis=1)
    edgelength = np.where(tvlen<1, 1.0, tvlen)

    data_sums = np.zeros(fastverts.shape[0], dtype=np.float) 
    connections = np.zeros(fastverts.shape[0], dtype=np.float) 

    # longer the edge distance to datapoint, less it has influence

    for _ in range(iterations):
        # step 1
        data_sums = np.zeros(data_sums.shape)
        connections = np.zeros(connections.shape)

        per_vert = data[edge_b]/edgelength
        safe_bincount(edge_a, per_vert, data_sums, connections)
        eb_smooth = data_sums/connections
        
        per_vert = eb_smooth[edge_a]/edgelength
        safe_bincount(edge_b, per_vert, data_sums, connections)

        new_data = data_sums/connections

        # step 2
        data_sums = np.zeros(data_sums.shape)
        connections = np.zeros(connections.shape)

        per_vert = data[edge_a]/edgelength
        safe_bincount(edge_b, per_vert, data_sums, connections)
        ea_smooth = data_sums/connections
        
        per_vert = ea_smooth[edge_b]/edgelength
        safe_bincount(edge_a, per_vert, data_sums, connections)

        new_data += data_sums/connections

        # limit between -1 and 1
        #new_data /= np.max([np.amax(new_data), np.abs(np.amin(new_data))])
        #data = new_data

    return new_data


def write_colors(values,  mesh):
    # Use 'curvature' vertex color entry for results
    if "Curvature" not in mesh.vertex_colors:
        mesh.vertex_colors.new(name="Curvature")
        
    color_layer = mesh.vertex_colors['Curvature']
    mesh.vertex_colors["Curvature"].active = True

    print("writing vertex colors for array:", values.shape)

    # write vertex colors
    mloops = np.zeros((len(mesh.loops)), dtype=np.int)
    mesh.loops.foreach_get("vertex_index", mloops)
    color_layer.data.foreach_set("color", values[mloops].flatten())


class CurvatureOperator(bpy.types.Operator):
    """Curvature to vertex colors"""
    bl_idname = "object.vertex_colors_curve"
    bl_label = "Curvature to vertex colors"
    bl_options = {'REGISTER', 'UNDO'}

    typesel = bpy.props.EnumProperty(
        items=[
            ("RED", "Red/Green", "", 1),
            ("GREY", "Grayscale", "", 2),
            ("GREYC", "Grayscale combined", "", 3),
            ("DIRECTION", "Direction", "", 4),
            ("THRESHOLD", "Threshold", "", 5),
            ("LAPLACIAN", "Laplace-Beltrami", "", 6),
            ],
        name="Output style",
        default="LAPLACIAN")
        
    concavity = bpy.props.BoolProperty(
        name="Concavity",
        default=True,
        options={'HIDDEN'})
    convexity = bpy.props.BoolProperty(
        name="Convexity",
        default=True,
        options={'HIDDEN'})
    
    def curveUpdate(self, context):
        if self.curvesel == "CAVITY":
            self.concavity = True
            self.convexity = False
        if self.curvesel == "VEXITY":
            self.concavity = False
            self.convexity = True
        if self.curvesel == "BOTH":
            self.concavity = True
            self.convexity = True
    
    curvesel = bpy.props.EnumProperty(
        items=[
            ("CAVITY", "Concave", "", 1),
            ("VEXITY", "Convex", "", 2),
            ("BOTH", "Both", "", 3),
            ],
        name="Curvature type",
        default="BOTH",
        update=curveUpdate)
    
    intensity_multiplier = bpy.props.FloatProperty(
        name="Intensity Multiplier",
        min=0.0,
        default=1.0)
        
    smooth = bpy.props.IntProperty(
        name="Smoothing steps",
        min=0,
        max=200,
        default=2)

    invert = bpy.props.BoolProperty(
        name="Invert",
        default=False)

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.mode == 'OBJECT'

    def set_colors(self, mesh, fvals):
        retvalues = np.ones((len(fvals), 4))
        
        if self.typesel == "GREY":
            splitter = fvals>0.5
            a_part = splitter * (fvals*2-1)*self.concavity
            b_part = np.logical_not(splitter) * (1-fvals*2)*self.convexity
            fvals = a_part + b_part
            fvals *= self.intensity_multiplier
            if self.invert:
                fvals = 1.0 - fvals
            
            retvalues[:,0] = fvals
            retvalues[:,1] = fvals
            retvalues[:,2] = fvals
            
        if self.typesel == "GREYC":
            if not self.convexity:
                fvals = np.where(fvals<0.5, 0.5, fvals)
            if not self.concavity:
                fvals = np.where(fvals>0.5, 0.5, fvals)
            if not self.invert:
                fvals = 1.0 - fvals
            fvals = (fvals-0.5)*self.intensity_multiplier+0.5
            retvalues[:,0] = fvals
            retvalues[:,1] = fvals
            retvalues[:,2] = fvals
            
        if self.typesel == "RED":
            splitter = fvals>0.5
            a_part = splitter * (fvals*2-1)*self.concavity
            b_part = np.logical_not(splitter) * (1-fvals*2)*self.convexity
            if self.invert:
                retvalues[:,0] = 1.0 - a_part * self.intensity_multiplier
                retvalues[:,1] = 1.0 - b_part * self.intensity_multiplier
            else:
                retvalues[:,0] = a_part * self.intensity_multiplier
                retvalues[:,1] = b_part * self.intensity_multiplier            
            retvalues[:,2] = np.zeros((len(fvals)))

        write_colors(retvalues, mesh)


    def execute(self, context):               
        mesh = context.active_object.data
        fastverts = read_verts(mesh)
        fastedges = read_edges(mesh)
        fastnorms = read_norms(mesh) 

        values = calc_normals(fastverts, fastnorms, fastedges)
        if self.smooth > 0:
            values = mesh_smooth_filter_variable(values, fastverts, fastedges, self.smooth)

        if self.typesel == "DIRECTION":
            values = average_parameter_vector(values, fastverts, fastedges) * self.intensity_multiplier
            values = (values+1)/2
            colors = np.ones((len(values), 4)) 
            colors[:,0] = values[:,0]
            colors[:,1] = values[:,1]
            colors[:,2] = values[:,2]
            write_colors(colors, mesh)

        elif self.typesel == "THRESHOLD":
            values2 = calc_normals(fastverts, fastnorms, fastedges)
            if self.smooth > 0:
                values2 = mesh_smooth_filter_variable(values2, fastverts, fastedges, self.smooth * int(2.0*self.intensity_multiplier))

            # make into 0=black 1=white
            values2 = 1.0-values2 
            values  = 1.0-values 
            
            values = np.where(values > 0.52, values-values2, 0.0)            
            values = np.where(values > 0.0, 1.0, 0.0)
            colors = np.ones((len(values), 4)) 
            colors[:,0] = values
            colors[:,1] = values
            colors[:,2] = values
            write_colors(colors, mesh)

        elif self.typesel == "LAPLACIAN":
            bpy.ops.object.mode_set(mode='EDIT')
            # K, k1, k2, H = laplace_beltrami(mesh)

            # print("Gaussian:",np.min(K), np.max(K))
            # print("H:",np.min(H), np.max(H))
            # print("K1:",np.min(k1), np.max(k1))
            # print("K2:",np.min(k2), np.max(k2))

            colors = np.ones((len(values), 4)) 
            # colors[:,0] = gaussian
            # colors[:,1] = gaussian
            # colors[:,2] = gaussian


            # k1 = H - sqrt(H**2 - K)
            # k2 = H + sqrt(H**2 - K)

            # divergence of surface normals is -2Hn => H = divg / (-2n)
            H = calc_normals(fastverts, fastnorms, fastedges)
            H = mesh_smooth_filter_variable(H, fastverts, fastedges, 1)
            K = edgewise_gaussian(mesh)
            K = mesh_smooth_filter_variable(K, fastverts, fastedges, 1)

            if np.min(H**2) < np.max(K):
                print("WARNING: normalized Gaussian")
                print("Gaussian:",np.min(K), np.max(K))
                max_K = np.min(H**2)
                K /= np.max(K)
                K *= max_K

            k1 = H + np.sqrt(H**2 - K)
            k2 = H - np.sqrt(H**2 - K)
            
            print("Gaussian:",np.min(K), np.max(K))
            print("H:",np.min(H), np.max(H))
            print("K1:",np.min(k1), np.max(k1))
            print("K2:",np.min(k2), np.max(k2))

            K -= np.min(K)
            K /= np.max(K)
            K *= self.intensity_multiplier

            H -= np.min(H)
            H /= np.max(H)
            H *= self.intensity_multiplier 

            k1 -= np.min(k1)
            k1 /= np.max(k1)
            k2 -= np.min(k2)
            k2 /= np.max(k2)

            k1 *= self.intensity_multiplier
            k2 *= self.intensity_multiplier

            colors[:,0] = K
            colors[:,1] = 0
            colors[:,2] = H
            bpy.ops.object.mode_set(mode='OBJECT')
            write_colors(colors, mesh)
        else:
            self.set_colors(mesh, values)           
                
        return {'FINISHED'}


def add_object_button(self, context):  
    self.layout.operator(  
        CurvatureOperator.bl_idname,  
        text=CurvatureOperator.__doc__,  
        icon='MESH_DATA')  


def register():
    bpy.utils.register_class(CurvatureOperator)
    bpy.types.VIEW3D_MT_object.append(add_object_button) 


def unregister():
    bpy.utils.unregister_class(CurvatureOperator)
    bpy.types.VIEW3D_MT_object.remove(add_object_button)


def profile_debug():
    pr = cProfile.Profile()
    pr.enable()
    bpy.ops.object.vertex_colors_curve()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats(sortby).print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    #unregister()
    register()
    #profile_debug()

