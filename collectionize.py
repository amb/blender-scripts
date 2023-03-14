import bpy
import mathutils as mu
from collections import defaultdict

bl_info = {
    "name": "Collectionize",
    "author": "ambi",
    "version": (1, 0),
    "blender": (3, 1, 0),
    "location": "View3D > Object > Collectionize",
    "description": "Create collections from object names",
    "category": "Object",
}

# This is pretty much what delete does, maybe later make it do more

# class FlattenCollection(bpy.types.Operator):
#     """Unlink all objects from selected collection and its children, and link
#     them to the parent collection, if any"""

#     bl_idname = "object.flatten_collection"
#     bl_label = "Destroy collection, link all objects to scene"
#     bl_options = {"REGISTER", "UNDO"}

#     def execute(self, context):
#         col = context.collection
#         if col is None:
#             self.report({"WARNING"}, "No collection selected")
#             return {"CANCELLED"}

#         parent = context.scene.collection

#         for o in col.all_objects:
#             if o.name in col.objects:
#                 col.objects.unlink(o)
#             parent.objects.link(o)

#         return {"FINISHED"}


def get_collection_and_objects(self, context):
    # create a list of all selected objects
    selected_objects = [o for o in context.selected_objects]
    if not selected_objects:
        # if no objects are selected, use all objects in the active collection
        selected_objects = [o for o in context.collection.objects]
        if selected_objects:
            self.report({"WARNING"}, "No objects selected, using all objects in active collection")
        else:
            self.report({"WARNING"}, "No objects or collections selected")
            return None, None

    # get collection of selected object
    parent_collection = selected_objects[0].users_collection[0]

    # if all objects are in the same collection, use that as parent
    for o in selected_objects:
        if o.users_collection[0] != parent_collection:
            parent_collection = None
            break

    if parent_collection is None:
        parent_collection = context.scene.collection

    return parent_collection, selected_objects


class CreateHierarchy(bpy.types.Operator):
    """Create object hierarchy with empties from object names, using a separator character
    to indicate hierarchy"""

    bl_idname = "object.empty_hierarchy_from_names"
    bl_label = "Create hierarchy from object names"
    bl_options = {"REGISTER", "UNDO"}

    separator: bpy.props.StringProperty(
        name="Separator",
        description="Separator character",
        default="_",
        maxlen=1,
    )
    depth: bpy.props.IntProperty(
        name="Depth",
        description="Maximum depth of hierarchy",
        default=3,
        min=1,
        max=10,
    )
    move_to_location: bpy.props.BoolProperty(
        name="Move to center",
        description="Move all empties to the first contained object's location",
        default=True,
    )

    def execute(self, context):
        parent_collection, selected_objects = get_collection_and_objects(self, context)
        if selected_objects is None:
            return {"CANCELLED"}
        empties = {}
        for o in selected_objects:
            # skip all objects that already have a parent
            if o.parent is not None:
                continue

            parts = o.name.split(self.separator)
            parent = None
            for i in range(self.depth):
                if i >= len(parts):
                    break

                if parts[i] not in empties:
                    e = bpy.data.objects.new(parts[i], None)
                    e.parent = parent
                    empties[parts[i]] = e
                    parent_collection.objects.link(e)

                parent = empties[parts[i]]

            o.parent = parent

        # move all empties to the center of the contained objects
        if self.move_to_location:
            for k, v in empties.items():
                if v.parent is None or len(v.children) == 0:
                    continue
                # csum = sum([o.location for o in v.children], mu.Vector((0, 0, 0)))
                # / len(v.children)
                # empties[k].location = csum
                csum = v.children[0].location.copy()
                empties[k].location = csum
                for c in v.children:
                    c.location -= csum

        return {"FINISHED"}


class Collectionize(bpy.types.Operator):
    """Create collections from object names, using a separator character to indicate hierarchy"""

    bl_idname = "object.collectionize"
    bl_label = "Collectionize"
    bl_options = {"REGISTER", "UNDO"}

    separator: bpy.props.StringProperty(
        name="Separator",
        description="Separator character",
        default="_",
        maxlen=1,
    )
    depth: bpy.props.IntProperty(
        name="Depth",
        description="Depth of collection hierarchy",
        default=1,
        min=0,
        max=10,
    )

    def execute(self, context):
        parent_collection, selected_objects = get_collection_and_objects(self, context)
        if selected_objects is None:
            return {"CANCELLED"}
        collections = defaultdict(list)
        for o in selected_objects:
            parts = o.name.split(self.separator)
            if self.depth >= len(parts):
                return {"CANCELLED"}

            collections[parts[self.depth]].append(o)

        for k, v in collections.items():
            col = bpy.data.collections.new(k)
            parent_collection.children.link(col)
            for o in v:
                if o.name in parent_collection.objects:
                    parent_collection.objects.unlink(o)
                col.objects.link(o)

        return {"FINISHED"}


def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(Collectionize.bl_idname)
    self.layout.operator(CreateHierarchy.bl_idname)


# add operator to outliner context menu
def outliner_menu_func(self, context):
    self.layout.separator()
    self.layout.operator(Collectionize.bl_idname)


def register():
    bpy.utils.register_class(Collectionize)
    bpy.utils.register_class(CreateHierarchy)
    bpy.types.VIEW3D_MT_object.append(menu_func)
    bpy.types.OUTLINER_MT_collection.append(outliner_menu_func)


def unregister():
    bpy.utils.unregister_class(Collectionize)
    bpy.utils.unregister_class(CreateHierarchy)
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    bpy.types.OUTLINER_MT_collection.remove(outliner_menu_func)
