# Copyright 2016 Tommi Hyppänen
# License: GPL 2

bl_info = {
    "name": "Brush Texture Autoload",
    "category": "Paint",
    "description": "Autoloading of brush textures from a folder",
    "author": "Tommi Hyppänen",
    "location": "3D view> Tool Shelf > Tools > Texture Autoload",
    "documentation": "community",
    "version": (0, 0, 1),
    "blender": (2, 76, 0)
}

import bpy
import os    
    
class AlphasLoadOperator(bpy.types.Operator):
    """Alpha Autoloading Operator"""
    bl_idname = "texture.alphas_batch_load"
    bl_label = "Autoload Texture Alphas"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        print("path:"+context.scene.alphas_location)
        print("abspath:"+bpy.path.abspath(context.scene.alphas_location))
        mypath = bpy.path.abspath(context.scene.alphas_location)
        dirfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        oktypes = set([".png", ".jpg"])
        okfiles = [f for f in dirfiles if f[-4:].lower() in oktypes]
        print(okfiles)
        
        for item in okfiles:
            fullname = os.path.join(mypath, item)
            tex = bpy.data.textures.new(item[:50]+'.autoload', type='IMAGE')
            tex.image = bpy.data.images.load(fullname)
            tex.use_alpha = True
        
        return {'FINISHED'}
    
class AlphasRemoveAllOperator(bpy.types.Operator):
    """Alpha Autoloading Remove All Operator"""
    bl_idname = "texture.alphas_batch_removeall"
    bl_label = "Autoremove Texture Alphas"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        remove_these = [i for i in bpy.data.textures.keys() if 'autoload' in i.split('.')]
        for item in remove_these:
            if not bpy.data.textures[item].users:
                bpy.data.textures.remove(bpy.data.textures[item])
        
        return {'FINISHED'}
    
class TextureAutoloadPanel(bpy.types.Panel):
    """Creates a Panel for Texture Autoload addon"""
    bl_label = "Texture Autoload"
    bl_idname = "OBJECT_PT_hello"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Tools"
    bl_context = "imagepaint"

    def draw(self, context):
        layout = self.layout

        obj = context.object

        row = layout.row()
        row.prop(context.scene, "alphas_location", text="Alphas Location")
        row = layout.row()
        row.operator(AlphasLoadOperator.bl_idname)
        row = layout.row()
        row.operator(AlphasRemoveAllOperator.bl_idname)

class_names = [AlphasLoadOperator, AlphasRemoveAllOperator, TextureAutoloadPanel]

def register():    
    bpy.utils.register_class(AlphasLoadOperator)
    bpy.utils.register_class(AlphasRemoveAllOperator)
    bpy.utils.register_class(TextureAutoloadPanel)
    bpy.types.Scene.alphas_location = bpy.props.StringProperty(
        name="alphas_path",
        description="Alphas Location",
        subtype='DIR_PATH')

def unregister():
    bpy.utils.unregister_class(AlphasLoadOperator)
    bpy.utils.unregister_class(AlphasRemoveAllOperator)
    bpy.utils.unregister_class(TextureAutoloadPanel)
    del bpy.types.Scene.alphas_location


if __name__ == "__main__":
    register()
