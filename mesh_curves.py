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
    "version": (0, 1, 3),
    "blender": (2, 74, 0)
}

import bpy
import random
from collections import defaultdict
import mathutils
import math
import numpy as np
import cProfile, pstats, io

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
        default=12.0)
        
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
            
            retvalues = np.zeros((len(fvals), 3))
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
            retvalues = np.zeros((len(fvals), 3))
            retvalues[:,0] = fvals
            retvalues[:,1] = fvals
            retvalues[:,2] = fvals
            
        if self.typesel == "RED":
            splitter = fvals>0.5
            a_part = splitter * (fvals*2-1)*self.concavity
            b_part = np.logical_not(splitter) * (1-fvals*2)*self.convexity
            retvalues = np.zeros((len(fvals), 3))
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
    
    def read_verts(self, mesh):
        mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
        mesh.vertices.foreach_get("co", mverts_co)
        return np.reshape(mverts_co, (len(mesh.vertices), 3))      
    
    def read_edges(self, mesh):
        fastedges = np.zeros((len(mesh.edges)*2), dtype=np.int) # [0.0, 0.0] * len(mesh.edges)
        mesh.edges.foreach_get("vertices", fastedges)
        return np.reshape(fastedges, (len(mesh.edges), 2))
    
    def read_norms(self, mesh):
        mverts_no = np.zeros((len(mesh.vertices)*3), dtype=np.float)
        mesh.vertices.foreach_get("normal", mverts_no)
        return np.reshape(mverts_no, (len(mesh.vertices), 3))
                
    def execute(self, context):               
        mesh = context.active_object.data
        fastverts = self.read_verts(mesh)
        fastedges = self.read_edges(mesh)
        fastnorms = self.read_norms(mesh)

        # Map the other vertex on an edge to vertex index
        # connected_verts = [[]] * len(fastverts) # FIXME: for some reason this is broken!?
        connected_verts = []
        for i in range(len(fastverts)):
            connected_verts.append([])

        for i in range(len(fastedges)):
            edge = fastedges[i][0], fastedges[i][1]
            connected_verts[edge[0]].append(edge[1])
            connected_verts[edge[1]].append(edge[0])    
 
        # Main calculation
        vertcount = len(mesh.vertices)
        angvalues = np.zeros(vertcount, dtype=np.float)

        multiplier = 100
        minedge = 1
        maxverts = 5
        dotps = np.zeros(maxverts)
        for i in range(vertcount):
            dotps.fill(0)
            psi = 0
            for v in connected_verts[i]:
                tvec = fastverts[v]-fastverts[i]
                tvlen = np.sqrt(tvec[0]*tvec[0] + tvec[1]*tvec[1] + tvec[2]*tvec[2])
                thisdot = (tvec/tvlen).dot(fastnorms[i])
                edgelength = tvlen * multiplier
                if edgelength < minedge:
                    edgelength = minedge
                dotps[psi] = thisdot/edgelength
                if psi<maxverts-1:
                    psi+=1
            # sum results
            angvalues[i] = 1.0 - np.arccos(np.add.reduce(dotps)/(psi+1))/np.pi

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
