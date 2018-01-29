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
import mathutils
import math
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


def safe_bincount(data, weights, dts, conn):
    bc = np.bincount(data, weights)
    dts[:len(bc)] += bc
    bc = np.bincount(data)
    conn[:len(bc)] += bc
    return (dts, conn)


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
            ],
        name="Output style",
        default="RED")
        
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
        # Use 'curvature' vertex color entry for results
        if "Curvature" not in mesh.vertex_colors:
            mesh.vertex_colors.new(name="Curvature")
            
        color_layer = mesh.vertex_colors['Curvature']
        mesh.vertex_colors["Curvature"].active = True

        retvalues = []
        
        if self.typesel == "GREY":
            splitter = fvals>0.5
            a_part = splitter * (fvals*2-1)*self.concavity
            b_part = np.logical_not(splitter) * (1-fvals*2)*self.convexity
            fvals = a_part + b_part
            fvals *= self.intensity_multiplier
            if self.invert:
                fvals = 1.0 - fvals
            
            retvalues = np.ones((len(fvals), 4))
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
            retvalues = np.ones((len(fvals), 4))
            retvalues[:,0] = fvals
            retvalues[:,1] = fvals
            retvalues[:,2] = fvals
            
        if self.typesel == "RED":
            splitter = fvals>0.5
            a_part = splitter * (fvals*2-1)*self.concavity
            b_part = np.logical_not(splitter) * (1-fvals*2)*self.convexity
            retvalues = np.ones((len(fvals), 4))
            if self.invert:
                retvalues[:,0] = 1.0 - a_part * self.intensity_multiplier
                retvalues[:,1] = 1.0 - b_part * self.intensity_multiplier
            else:
                retvalues[:,0] = a_part * self.intensity_multiplier
                retvalues[:,1] = b_part * self.intensity_multiplier            
            retvalues[:,2] = np.zeros((len(fvals)))

        # write vertex colors
        mloops = np.zeros((len(mesh.loops)), dtype=np.int)
        mesh.loops.foreach_get("vertex_index", mloops)
        color_layer.data.foreach_set("color", retvalues[mloops].flatten())
        
        return None


    def calc_normals(self, mesh, fastverts, fastnorms, fastedges):
        # FIXME: FAILS AT INVALID INPUT MESH
        #        If there are any loose or disconnected vertices or edges, the output will be black
        # HOWTO cleanup:
        #        1. Remove doubles
        #        2. Delete loose

        edge_a, edge_b = fastedges[:,0], fastedges[:,1]
        
        tvec = fastverts[edge_b] - fastverts[edge_a]
        tvlen = np.linalg.norm(tvec, axis=1)    

        tvec = (tvec.T / tvlen).T # normalize vectors

        # adjust the minimum of what is processed   
        edgelength = tvlen * 100 
        edgelength = np.where(edgelength<1, 1.0, edgelength)

        vecsums = np.zeros(fastverts.shape[0], dtype=np.float) 
        connections = np.zeros(fastverts.shape[0], dtype=np.float) 

        # calculate normal differences to the edge vector in the first edge vertex
        totdot = (np.einsum('ij,ij->i', tvec, fastnorms[edge_a]))/edgelength
        #for i, v in enumerate(edge_a):
        #    vecsums[v] += totdot[i]
        #    connections[v] += 1
        safe_bincount(edge_a, totdot, vecsums, connections)

        # calculate normal differences to the edge vector  in the second edge vertex
        totdot = (np.einsum('ij,ij->i', -tvec, fastnorms[edge_b]))/edgelength
        safe_bincount(edge_b, totdot, vecsums, connections)

        # (approximate gaussian) curvature is the average difference of 
        # edge vectors to surface normals (from dot procuct cosine equation)
        curve = 1.0 - np.arccos(vecsums/connections)/np.pi

        # 1 = max curvature, 0 = min curvature, 0.5 = zero curvature
        curve -= 0.5
        curve /= np.max([np.amax(curve), np.abs(np.amin(curve))])
        curve += 0.5
        return curve
                
    def mesh_smooth_filter_variable(self, mesh, data, fastverts, fastedges):
        # vert indices of edges
        edge_a, edge_b = fastedges[:,0], fastedges[:,1]
        tvlen = np.linalg.norm(fastverts[edge_b] - fastverts[edge_a], axis=1)
        edgelength = np.where(tvlen<1, 1.0, tvlen)

        data_sums = np.zeros(fastverts.shape[0], dtype=np.float) 
        connections = np.zeros(fastverts.shape[0], dtype=np.float) 

        # longer the edge distance to datapoint, less it has influence

        # step 1
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
        new_data /= np.max([np.amax(new_data), np.abs(np.amin(new_data))])

        return new_data


    def execute(self, context):               
        mesh = context.active_object.data
        fastverts = read_verts(mesh)
        fastedges = read_edges(mesh)
        fastnorms = read_norms(mesh) 

        angvalues = self.calc_normals(mesh, fastverts, fastnorms, fastedges)
        if self.smooth > 0:
            angvalues -= 0.5
            angvalues *= 2.0
            for _ in range(self.smooth):
                angvalues = self.mesh_smooth_filter_variable(mesh, angvalues, fastverts, fastedges)
            angvalues /= 2.0
            angvalues += 0.5
        
        self.set_colors(mesh, angvalues)           
                
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

