import bpy
import bmesh
import math
import mathutils as mu
import numpy as np

from bpy.types import Header, Menu, Panel

OLD_DRAW = None
OLD_OBJECT_ADD = None

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

    bpy.types.Scene.new_objects_to_cursor = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.new_objects_size_multiplier = bpy.props.FloatProperty(default=1.0, min=0.01, max=200.0)

    # ----- HACK: 3D cursor panel additions -----
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

        layout.prop(cursor, "rotation_mode", text="")
        
        row = layout.row()
        row.operator(TFC_OT_TransformFromCursor.bl_idname, text="Copy to object")

        row = layout.row()
        row.prop(context.scene, "new_objects_to_cursor", text="Align new objects to cursor")
        if context.scene.new_objects_to_cursor is True:
            row = layout.row()
            row.prop(context.scene, "new_objects_size_multiplier", text="Size")



    bpy.types.VIEW3D_PT_view3d_cursor.draw = draw

    # ----- HACK: Object add menu additions -----
    global OLD_OBJECT_ADD
    if OLD_OBJECT_ADD == None:
        OLD_OBJECT_ADD = bpy.types.VIEW3D_MT_mesh_add.draw

    def draw(self, context):
        layout = self.layout

        layout.operator_context = "INVOKE_REGION_WIN"

        if context.scene.new_objects_to_cursor is True:
            op = layout.operator("mesh.primitive_plane_add", text="Plane", icon="MESH_PLANE")
            op.rotation = context.scene.cursor.rotation_euler
            op.size = 2.0 * context.scene.new_objects_size_multiplier
            op = layout.operator("mesh.primitive_cube_add", text="Cube", icon="MESH_CUBE")
            op.rotation = context.scene.cursor.rotation_euler
            op.size = 2.0 * context.scene.new_objects_size_multiplier
            op = layout.operator("mesh.primitive_circle_add", text="Circle", icon="MESH_CIRCLE")
            op.rotation = context.scene.cursor.rotation_euler
            op.radius = 1.0 * context.scene.new_objects_size_multiplier
            op = layout.operator("mesh.primitive_uv_sphere_add", text="UV Sphere", icon="MESH_UVSPHERE")
            op.rotation = context.scene.cursor.rotation_euler
            op.radius = 1.0 * context.scene.new_objects_size_multiplier
            op = layout.operator("mesh.primitive_ico_sphere_add", text="Ico Sphere", icon="MESH_ICOSPHERE")
            op.rotation = context.scene.cursor.rotation_euler
            op.radius = 1.0 * context.scene.new_objects_size_multiplier
            op = layout.operator("mesh.primitive_cylinder_add", text="Cylinder", icon="MESH_CYLINDER")
            op.rotation = context.scene.cursor.rotation_euler
            op.radius = 1.0 * context.scene.new_objects_size_multiplier
            op.depth = 2.0 * context.scene.new_objects_size_multiplier
            op = layout.operator("mesh.primitive_cone_add", text="Cone", icon="MESH_CONE")
            op.rotation = context.scene.cursor.rotation_euler
            op.radius1 = 1.0 * context.scene.new_objects_size_multiplier
            op.depth = 2.0 * context.scene.new_objects_size_multiplier
            op = layout.operator("mesh.primitive_torus_add", text="Torus", icon="MESH_TORUS")
            op.rotation = context.scene.cursor.rotation_euler
            # op.size = 2.0 * context.scene.new_objects_size_multiplier

            layout.separator()

            op = layout.operator("mesh.primitive_grid_add", text="Grid", icon="MESH_GRID")
            op.rotation = context.scene.cursor.rotation_euler
            op.size = 2.0 * context.scene.new_objects_size_multiplier
            op = layout.operator("mesh.primitive_monkey_add", text="Monkey", icon="MESH_MONKEY")
            op.rotation = context.scene.cursor.rotation_euler
            op.size = 2.0 * context.scene.new_objects_size_multiplier
        else:
            op = layout.operator("mesh.primitive_plane_add", text="Plane", icon="MESH_PLANE")
            op = layout.operator("mesh.primitive_cube_add", text="Cube", icon="MESH_CUBE")
            op = layout.operator("mesh.primitive_circle_add", text="Circle", icon="MESH_CIRCLE")
            op = layout.operator("mesh.primitive_uv_sphere_add", text="UV Sphere", icon="MESH_UVSPHERE")
            op = layout.operator("mesh.primitive_ico_sphere_add", text="Ico Sphere", icon="MESH_ICOSPHERE")
            op = layout.operator("mesh.primitive_cylinder_add", text="Cylinder", icon="MESH_CYLINDER")
            op = layout.operator("mesh.primitive_cone_add", text="Cone", icon="MESH_CONE")
            op = layout.operator("mesh.primitive_torus_add", text="Torus", icon="MESH_TORUS")

            layout.separator()

            op = layout.operator("mesh.primitive_grid_add", text="Grid", icon="MESH_GRID")
            op = layout.operator("mesh.primitive_monkey_add", text="Monkey", icon="MESH_MONKEY")

    bpy.types.VIEW3D_MT_mesh_add.draw = draw


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

    del bpy.types.Scene.new_objects_to_cursor
    del bpy.types.Scene.new_objects_size_multiplier

    global OLD_DRAW
    if OLD_DRAW is not None:
        bpy.types.VIEW3D_PT_view3d_cursor.draw = OLD_DRAW

    global OLD_OBJECT_ADD
    if OLD_OBJECT_ADD is not None:
        bpy.types.VIEW3D_MT_mesh_add.draw = OLD_OBJECT_ADD


if __name__ == "__main__":
    register()
