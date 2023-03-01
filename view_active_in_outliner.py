import bpy

bl_info = {
    "category": "Object",
    "name": "View active object in outliner",
    "blender": (3, 1, 0),
    "author": "ambi",
    "version": (1, 0),
    "location": "View3D > Object > View active object in outliner",
    "description": "View active object in outliner",
    "warning": "Experimental",
    "addon_support": "TESTING",
}


def view_act_object_in_outliner():
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "OUTLINER":
                region = next((r for r in area.regions if r.type == "WINDOW"))
                if region is not None:
                    m = {"area": area, "region": region}
                    bpy.ops.outliner.show_active(m)


class ViewActiveObjectInOutliner(bpy.types.Operator):
    """View active object in outliner"""

    bl_idname = "object.view_active_object_in_outliner"
    bl_label = "View active object in outliner"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        view_act_object_in_outliner()
        return {"FINISHED"}


def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(
        ViewActiveObjectInOutliner.bl_idname, text=ViewActiveObjectInOutliner.bl_label
    )


def register():
    bpy.utils.register_class(ViewActiveObjectInOutliner)
    bpy.types.VIEW3D_MT_object_context_menu.append(menu_func)


def unregister():
    bpy.utils.unregister_class(ViewActiveObjectInOutliner)
    bpy.types.VIEW3D_MT_object_context_menu.remove(menu_func)
