import bpy
import bmesh
import math
import mathutils as mu
import numpy as np

OLD_DRAW = None

bl_info = {
    "name": "Transform from cursor",
    "author": "Tommi Hyppänen (ambi)",
    "version": (1, 0, 1),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > View > 3D Cursor",
    "description": "Copy cursor rotation to object transform.",
    "category": "Object",
}


class TFC_OT_TransformFromCursor(bpy.types.Operator):
    bl_idname = "object.transform_from_cursor"
    bl_label = "Transform from cursor"
    bl_description = "Copy 3D cursor rotation to object transform rotation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mode = context.object.mode
        bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        obj = context.active_object

        to_qt = context.scene.cursor.rotation_euler.to_quaternion()

        to_qt.invert()
        context.active_object.rotation_euler = to_qt.to_euler()
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        to_qt.invert()

        context.active_object.rotation_euler = to_qt.to_euler()

        bpy.ops.object.mode_set(mode=mode)

        return {"FINISHED"}


classes = (TFC_OT_TransformFromCursor,)


def register():
    for c in classes:
        bpy.utils.register_class(c)

    global OLD_DRAW
    if OLD_DRAW == None:
        OLD_DRAW = bpy.types.VIEW3D_PT_view3d_cursor.draw

    def draw(self, context):
        layout = self.layout

        cursor = context.scene.cursor

        layout.column().prop(cursor, "location", text="Location")
        rotation_mode = cursor.rotation_mode
        if rotation_mode == "QUATERNION":
            layout.column().prop(cursor, "rotation_quaternion", text="Rotation")
        elif rotation_mode == "AXIS_ANGLE":
            layout.column().prop(cursor, "rotation_axis_angle", text="Rotation")
        else:
            layout.column().prop(cursor, "rotation_euler", text="Rotation")

        row = layout.row()
        row.operator(TFC_OT_TransformFromCursor.bl_idname, text="Copy to object")

        layout.prop(cursor, "rotation_mode", text="")

    bpy.types.VIEW3D_PT_view3d_cursor.draw = draw


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    global OLD_DRAW
    bpy.types.VIEW3D_PT_view3d_cursor.draw = OLD_DRAW


if __name__ == "__main__":
    register()
