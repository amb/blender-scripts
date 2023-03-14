bl_info = {
    "name": "Fix Objects with Negative Scale",
    "author": "ambi",
    "version": (1, 0),
    "blender": (3, 1, 0),
    "location": "View3D > Object > Select > Fix Negative Scale",
    "description": "Selects all objects in the scene with negative scale and applies the transform",
    "category": "Object",
}

import bpy


class OBJECT_OT_fix_negative_scale(bpy.types.Operator):
    bl_idname = "object.fix_negative_scale"
    bl_label = "Fix Negative Scale"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mirrored_mesh = {}
        objs = bpy.context.scene.objects

        for obj in objs:
            count = int(obj.scale.x < 0) + int(obj.scale.y < 0) + int(obj.scale.z < 0)
            if count % 2 == 1:
                if obj.data not in mirrored_mesh:
                    new_mesh = obj.data.copy()
                    for v in new_mesh.vertices:
                        v.co.x *= -1.0
                    mirrored_mesh[obj.data] = new_mesh

        for obj in objs:
            count = int(obj.scale.x < 0) + int(obj.scale.y < 0) + int(obj.scale.z < 0)
            if count % 2 == 1:
                obj.data = mirrored_mesh[obj.data]
                obj.scale.x = -obj.scale.x
                obj.select_set(True)
            else:
                obj.select_set(False)

        if len(mirrored_mesh) > 0:
            self.report({"INFO"}, "Created new meshes: %d" % len(mirrored_mesh))
        else:
            self.report({"INFO"}, "No negative scale objects found")

        return {"FINISHED"}


def menu_func(self, context):
    self.layout.operator(OBJECT_OT_fix_negative_scale.bl_idname)


def register():
    bpy.utils.register_class(OBJECT_OT_fix_negative_scale)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_fix_negative_scale)
    bpy.types.VIEW3D_MT_object.remove(menu_func)


if __name__ == "__main__":
    register()
