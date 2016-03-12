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
    "name": "Triplanar UV mapping",
    "category": "UV",
    "description": "Generate triplanar UV mapping from an object",
    "author": "Tommi Hyppänen (ambi)",
    "location": "Space bar quick menu > Triplanar UV mapping",
    "version": (0, 0, 1),
    "blender": (2, 76, 0)
}

import bpy
import bmesh
import math

if bpy.app.version[0] < 2 or bpy.app.version[1] < 62:
    raise Exception("This Triplanar UV mapping addons works only in Blender 2.62 and above")

def main(context):
    obj = context.active_object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    uv_layer = bm.loops.layers.uv.verify()
    bm.faces.layers.tex.verify()  # currently blender needs both layers.

    # adjust UVs
    for f in bm.faces:
        norm = f.normal
        ax, ay, az = abs(norm.x), abs(norm.y), abs(norm.z)
        axis = -1
        if ax > ay and ax > az:
            axis = 0
        if ay > ax and ay > az:
            axis = 1
        if az > ax and az > ay:
            axis = 2
        for l in f.loops:
            luv = l[uv_layer]
            if axis == 0: # x plane     
                luv.uv.x = l.vert.co.y
                luv.uv.y = l.vert.co.z
            if axis == 1: # u plane
                luv.uv.x = l.vert.co.x
                luv.uv.y = l.vert.co.z
            if axis == 2: # z plane
                luv.uv.x = l.vert.co.x
                luv.uv.y = l.vert.co.y

    bmesh.update_edit_mesh(me)


class UvOperator(bpy.types.Operator):
    """Triplanar UV mapping"""
    bl_idname = "uv.triplanar"
    bl_label = "Triplanar UV mapping"

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH')

    def execute(self, context):
        main(context)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(UvOperator)


def unregister():
    bpy.utils.unregister_class(UvOperator)


if __name__ == "__main__":
    register()
