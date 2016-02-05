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
        print(ax,ay,az)
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
