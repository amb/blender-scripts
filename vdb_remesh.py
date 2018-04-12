bl_info = {
    "name": "OpenVDB remesh",
    "category": "Object",
    "description": "OpenVDB based remesher",
    "author": "Tommi Hyppänen (ambi)",
    "location": "3D View > Retopology > VDB remesh",
    "version": (0, 1, 1),
    "blender": (2, 79, 0)
}

try:
    import pyopenvdb as vdb
except SystemError as e:
    print(e)
import numpy as np
import bpy
import bmesh
import cProfile, pstats, io

DEBUG = False 

def write_slow(mesh, ve, tr, qu):
    bm = bmesh.new()
    for co in ve.tolist():
        bm.verts.new(co)

    bm.verts.ensure_lookup_table()    

    for face_indices in tr.tolist() + qu.tolist():
        bm.faces.new(tuple(bm.verts[index] for index in face_indices))

    return bm


def write_fast(ve, tr, qu):
    # TODO: He's dead, Jim
    me = bpy.data.meshes.new("testmesh")

    quadcount = len(qu)
    tricount  = len(tr)

    me.vertices.add(count=len(ve))
    print(len(ve), ve.shape, len(me.vertices))
    me.vertices.foreach_set("co", ve.ravel())

    loopcount = quadcount * 4 + tricount * 3
    me.loops.add(loopcount)
    me.polygons.add(quadcount + tricount)
    face_lengths = np.empty(quadcount + tricount, dtype=np.int)
    face_lengths[:tricount] = 3
    face_lengths[quadcount:] = 4
    loops = np.concatenate((np.arange(tricount) * 3, np.arange(quadcount) * 4 + tricount * 3))
    print(loops.shape, face_lengths.shape)
    me.polygons.foreach_set("loop_total", face_lengths)
    me.polygons.foreach_set("loop_start", loops)
    v_out = np.concatenate((tr.ravel(), qu.ravel()))
    print(len(v_out), v_out.dtype, len(ve), len(me.polygons), loopcount)
    #print(v_out[:10], v_out[tricount*3-5:tricount*3+5])
    #print(qu[:10])
    print(len(me.vertices), np.max(v_out), np.max(loops))
    me.polygons.foreach_set("vertices", v_out)
    """
    count = 0
    for i, f in enumerate(me.polygons):
        #len(f.vertices)      
        for j, idx in enumerate(f.vertices):
            f.vertices[j] = v_out[count]
            count += 1
    """

    me.update(calc_edges=True)
    me.validate(verbose=True)

    ob = bpy.data.objects.new("testing", me)
    bpy.context.scene.objects.link(ob)

def read_verts(mesh):
    mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
    mesh.vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(mesh.vertices), 3))      

def read_loops(mesh):
    loops = np.zeros((len(mesh.polygons)), dtype=np.int)
    mesh.polygons.foreach_get("loop_total", loops)
    return loops 

def vdb_remesh(verts, tris, quads, iso, adapt, only_quads, vxsize, blur):
    vtransform = vdb.createLinearTransform(voxelSize=vxsize)
    iso *= vxsize
    
    if len(tris) == 0:
        grid = vdb.FloatGrid.createLevelSetFromPolygons(verts, quads=quads, transform=vtransform)
    elif len(quads) == 0:
        grid = vdb.FloatGrid.createLevelSetFromPolygons(verts, triangles=tris, transform=vtransform)
    else:
        grid = vdb.FloatGrid.createLevelSetFromPolygons(verts, tris, quads, transform=vtransform)

    if blur > 0:
        bb = grid.evalActiveVoxelBoundingBox()
        array = np.empty((bb[1][0]-bb[0][0], bb[1][1]-bb[0][1], bb[1][2]-bb[0][2]), dtype=np.float)
        grid.copyToArray(array, ijk=bb[0])
        for _ in range(blur):
            array = ( \
                np.roll(array, -1, axis=0) + np.roll(array, 1, axis=0) + \
                np.roll(array, -1, axis=1) + np.roll(array, 1, axis=1) + \
                np.roll(array, -1, axis=2) + np.roll(array, 1, axis=2) \
                ) / 6 
        grid.copyFromArray(array, ijk=bb[0])

    if only_quads:
        verts, quads = grid.convertToQuads(isovalue=iso)
        tris = np.array([], dtype=np.int)
    else:
        verts, tris, quads = grid.convertToPolygons(isovalue=iso, adaptivity=adapt)

    return (verts, tris, quads)

def read_mesh(mesh):
    verts = read_verts(mesh)

    qu, tr = [], []
    for f in mesh.polygons:
        if len(f.vertices) == 4:        
            qu.append([])
            for idx in f.vertices:
                qu[-1].append(idx)
        if len(f.vertices) == 3:        
            tr.append([])
            for idx in f.vertices:
                tr[-1].append(idx)

    return (verts, np.array(tr), np.array(qu))


def read_bmesh(bmesh):
    bmesh.verts.ensure_lookup_table()
    bmesh.faces.ensure_lookup_table()

    verts = [(i.co[0], i.co[1], i.co[2]) for i in bmesh.verts]
    qu, tr = [], []
    for f in bmesh.faces:
        if len(f.verts) == 4:        
            qu.append([])
            for v in f.verts:
                qu[-1].append(v.index)
        if len(f.verts) == 3:        
            tr.append([])
            for v in f.verts:
                tr[-1].append(v.index)

    return (np.array(verts), np.array(tr), np.array(qu))


def project_to_nearest(source, target):
    source.verts.ensure_lookup_table()   

    res = target.closest_point_on_mesh(source.verts[0].co)
    
    if len(res) > 3:
        # new style 2.79
        rnum = 1
    else:
        # old style 2.76
        rnum = 0

    for i, vtx in enumerate(source.verts):
        res = target.closest_point_on_mesh(vtx.co)
        #if res[0]:
        source.verts[i].co = res[rnum]
    

class VDBRemeshOperator(bpy.types.Operator):
    """OpenVDB Remesh"""
    bl_idname = "object.vdbremesh_operator"
    bl_label = "OpenVDB Remesh Operator"
    bl_options = {'REGISTER', 'UNDO'}

    voxel_size = bpy.props.FloatProperty(
            name="Voxel size",
            description="Voxel size",
            min=0.01, max=0.5,
            default=0.02)

    isosurface = bpy.props.FloatProperty(
            name="Isosurface",
            description="Isosurface",
            min=-3.0, max=3.0,
            default=0.0)

    adaptivity = bpy.props.FloatProperty(
            name="Adaptivity",
            description="Adaptivity",
            min=0.0, max=1.0,
            default=0.01)

    blur = bpy.props.IntProperty(
            name="Blur",
            description="Blur iterations",
            min=0, max=10,
            default=0)

    only_quads = bpy.props.BoolProperty(
            name="Only quads",
            description="Construct the mesh using only quad topology",
            default=False)

    smooth = bpy.props.BoolProperty(
            name="Smooth",
            description="Smooth shading toggle",
            default=True)

    project_nearest = bpy.props.BoolProperty(
            name="Project to nearest",
            description="Project generated mesh points to nearest surface point",
            default=False)


    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        if DEBUG:
            pr = cProfile.Profile()
            pr.enable()

        me = context.active_object.data
        bm = bmesh.new()
        bm.from_mesh(me)

        loops = read_loops(me)
        if np.max(loops) > 4:
            print("Mesh has ngons! Triangulating...")
            bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)

        startmesh = read_bmesh(bm)

        self.vert_0 = len(startmesh[0])

        new_mesh = vdb_remesh(startmesh[0], startmesh[1], startmesh[2], self.isosurface, \
            self.adaptivity, self.only_quads, self.voxel_size, self.blur)

        self.vert_1 = len(new_mesh[0])
        self.face_1 = len(new_mesh[1])+len(new_mesh[2])

        new_bm = write_slow(me, *new_mesh)

        if self.project_nearest:
            bpy.context.scene.update()
            project_to_nearest(new_bm, context.active_object)

        new_bm.to_mesh(me)
        new_bm.free()
        #write_fast(*new_mesh)

        bm.free()

        # recalc normals
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()

        if self.smooth:
            bpy.ops.object.shade_smooth()

        if DEBUG:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s)
            ps.strip_dirs().sort_stats(sortby).print_stats()
            print(s.getvalue())

        return {'FINISHED'}

    #def invoke(self, context, event):
    #    wm = context.window_manager
    #    return wm.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        col = layout.column()

        row = col.row()
        row.prop(self, "voxel_size")

        row = col.row()
        row.prop(self, "isosurface")

        row = col.row()
        row.prop(self, "adaptivity")

        row = col.row()
        row.prop(self, "blur")

        row = col.row()
        row.prop(self, "only_quads")
        row = col.row()
        row.prop(self, "smooth")
        row = col.row()
        row.prop(self, "project_nearest")

        if hasattr(self, 'vert_0'):
            infotext = "Change: {:.2%}".format(self.vert_1/self.vert_0)
            row = col.row()
            row.label(text=infotext)
            row = col.row()
            row.label(text="Verts: {}, Polys: {}".format(self.vert_1, self.face_1))

        

class VDBRemeshPanel(bpy.types.Panel):
    """OpenVDB remesh operator panel"""
    bl_label = "VDB remesh"
    bl_idname = "object.vdbremesh_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Retopology"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator(VDBRemeshOperator.bl_idname, text="OpenVDB remesh")


def register():
    bpy.utils.register_class(VDBRemeshOperator)
    bpy.utils.register_class(VDBRemeshPanel)


def unregister():
    bpy.utils.unregister_class(VDBRemeshOperator)
    bpy.utils.unregister_class(VDBRemeshPanel)


if __name__ == "__main__":
    register()
    #bpy.ops.object.vdbremesh_operator()


