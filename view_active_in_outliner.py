import bpy

bl_info = {
    "category": "Object",
    "name": "View active object in outliner",
    "blender": (3, 1, 0),
    "author": "ambi",
    "version": (1, 1),
    "location": "View3D > Object > View active object in outliner",
    "description": "View active object in outliner",
}

previous_selection = None
mouse_in_view3d = False


def view_act_object_in_outliner():
    global previous_selection
    if bpy.context.active_object != previous_selection:
        previous_selection = bpy.context.active_object
        bpy.ops.wm.mouse_position("INVOKE_DEFAULT")
        if mouse_in_view3d:
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == "OUTLINER":
                        region = next((r for r in area.regions if r.type == "WINDOW"))
                        if region is not None:
                            m = {"area": area, "region": region}
                            bpy.ops.outliner.show_active(m)
    return 0.1


def sync_toggle(self, context):
    if context.scene.sync_view3d_outliner:
        bpy.app.timers.register(view_act_object_in_outliner, first_interval=0.1)
    else:
        bpy.app.timers.unregister(view_act_object_in_outliner)


def is_in_view3d_area(x, y):
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            x_in_range = x >= area.x and x <= area.x + area.width
            y_in_range = y >= area.y and y <= area.y + area.height
            return x_in_range and y_in_range
    return False


class SimpleMouseOperator(bpy.types.Operator):
    bl_idname = "wm.mouse_position"
    bl_label = "Mouse location"

    x: bpy.props.IntProperty()
    y: bpy.props.IntProperty()

    def execute(self, context):
        global mouse_in_view3d
        mouse_in_view3d = is_in_view3d_area(self.x, self.y)
        # Mouse coords: self.x, self.y
        return {"FINISHED"}

    def invoke(self, context, event):
        self.x = event.mouse_x
        self.y = event.mouse_y
        return self.execute(context)


# create function to add a toggle button to the 3d view header
def draw_header(self, context):
    layout = self.layout
    if context.mode == "OBJECT":
        layout.label(text="Sync outliner")
        layout.prop(context.scene, "sync_view3d_outliner", text="")


def register():
    bpy.utils.register_class(SimpleMouseOperator)
    bpy.types.Scene.sync_view3d_outliner = bpy.props.BoolProperty(
        name="Sync View3D and Outliner",
        description="Sync View3D and Outliner",
        default=True,
        update=sync_toggle,
    )
    bpy.types.VIEW3D_HT_header.append(draw_header)
    bpy.app.timers.register(view_act_object_in_outliner, first_interval=1.0)


def unregister():
    bpy.utils.unregister_class(SimpleMouseOperator)
    bpy.types.VIEW3D_HT_header.remove(draw_header)
    del bpy.types.Scene.sync_view3d_outliner
    bpy.app.timers.unregister(view_act_object_in_outliner)
