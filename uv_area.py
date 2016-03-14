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
    "name": "UV Statistics",
    "category": "UV",
    "description": "Calculate relevant statistics on the active UV map",
    "author": "Tommi Hyppänen (ambi)",
    "location": "UV/Image Editor > Tools > Misc > UV Stats > Update Stats",
    "version": (0, 0, 5),
    "blender": (2, 76, 0)
}

# Calculate used UV area, works only for tri+quad meshes
# doesn't account for UV overlap
# not reliable for UV points closer to one another than 0.0002 units

import bpy
import math
import bmesh

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

def get_uv_area(ob, uv_layer):
    total_area = 0.0
    for poly in ob.data.polygons:
        if len(poly.loop_indices) == 3:
            total_area += triangle_area([uv_layer[i].uv for i in poly.loop_indices])
        if len(poly.loop_indices) == 4:
            total_area += quad_area([uv_layer[i].uv for i in poly.loop_indices])

    return total_area

def approximate_islands(ob, uv_layer):
    islands = []

    # merge polygons sharing uvs
    for poly in ob.data.polygons:
        uvs = set((round(uv_layer[i].uv[0], 4), round(uv_layer[i].uv[1], 4)) for i in poly.loop_indices)
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
        
class UVStatsOperator(bpy.types.Operator):
    """UV Statistics"""
    bl_idname = "uv.stats"
    bl_label = "UV Statistics"

    @classmethod
    def poll(cls, context):
        return bpy.context.active_object.mode == "EDIT" and len(context.object.data.uv_textures) > 0
        #return (context.object is not None and
        #        context.object.type == 'MESH' and
        #        len(context.object.data.uv_textures) > 0)

    def execute(self, context):
        prev_mode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        ob = bpy.context.object
        uv_layer = ob.data.uv_layers.active.data

        #me = ob.data
        #bm = bmesh.from_edit_mesh(me)

        #uv_layer = bm.loops.layers.uv.verify()
        #bm.faces.layers.tex.verify()
        
        #for f in bm.faces:
        #    for l in f.loops:
        #        luv = l[uv_layer]
        #        print(luv.uv)
        
        context.scene.uv_island_count = repr(len(approximate_islands(ob, uv_layer)))
        context.scene.uv_area_size = repr(round((get_uv_area(ob, uv_layer))*100, 2))+"%"

        print(context.scene.uv_island_count + " approximate UV islands counted.")
        print(context.scene.uv_area_size + " UV area used." )
        
        bpy.ops.object.mode_set(mode=prev_mode)
        return {'FINISHED'}
    

class UVStatsPanel(bpy.types.Panel):
    """UV Stats Panel"""
    bl_label = "UV Stats"
    bl_idname = "uv.statspanel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'TOOLS'
    #bl_category = "Tools"
    #bl_context = "imagepaint"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.label(text="Islands: "+context.scene.uv_island_count, icon='QUESTION')

        row = layout.row()
        row.label(text="Area: "+context.scene.uv_area_size, icon='QUESTION')

        row = layout.row()
        row.operator(UVStatsOperator.bl_idname, text="Update Stats")
        

def register():
    bpy.utils.register_class(UVStatsOperator)
    bpy.utils.register_class(UVStatsPanel)

    bpy.types.Scene.uv_island_count = bpy.props.StringProperty(
        name="uv_islands", description="UV Islands")

    bpy.types.Scene.uv_area_size = bpy.props.StringProperty(
        name="uv_area", description="UV Area")

    bpy.types.Scene.uv_distortion = bpy.props.StringProperty(
        name="uv_distortion", description="UV Distortion")

def unregister():
    bpy.utils.unregister_class(UVStatsOperator)
    bpy.utils.unregister_class(UVStatsPanel)
    

if __name__ == "__main__":
    register()
            