# Copyright 2018 Tommi Hyppänen, license: GNU General Public License v2.0

bl_info = {
    "name": "OpenVDB remesh",
    "category": "Object",
    "description": "OpenVDB based remesher",
    "author": "Tommi Hyppänen (ambi)",
    "location": "3D View > Retopology > VDB remesh",
    "version": (0, 1, 5),
    "blender": (2, 79, 0)
}

try:
    import pyopenvdb as vdb
except SystemError as e:
    print(e)
import numpy as np
import bpy
import bmesh
import random
import cProfile, pstats, io

DEBUG = True 
MIN_VOXEL_SIZE = 0.005
MAX_VOXEL_SIZE = 0.5
MAX_POLYGONS = 10000000

MAX_SMOOTHING = 1000*500*500
MAX_VOXEL_BB = 1500*1500*1500

def write_slow(mesh, ve, tr, qu):
    print("vdb_remesh: write mesh (slow)")
    bm = bmesh.new()
    for co in ve.tolist():
        bm.verts.new(co)

    bm.verts.ensure_lookup_table()    

    for face_indices in tr.tolist() + qu.tolist():
        bm.faces.new(tuple(bm.verts[index] for index in face_indices))

    return bm


def write_fast(ve, tr, qu):
    print("vdb_remesh: write mesh (fast)")
    me = bpy.data.meshes.new("testmesh")

    quadcount = len(qu)
    tricount  = len(tr)

    me.vertices.add(count=len(ve))

    loopcount = quadcount * 4 + tricount * 3
    facecount = quadcount + tricount
    
    me.loops.add(loopcount)
    me.polygons.add(facecount)

    face_lengths = np.zeros(facecount, dtype=np.int)
    face_lengths[:tricount] = 3
    face_lengths[tricount:] = 4

    loops = np.concatenate((np.arange(tricount) * 3, np.arange(quadcount) * 4 + tricount * 3))
    
    # [::-1] makes normals consistent (from OpenVDB)
    v_out = np.concatenate((tr.ravel()[::-1], qu.ravel()[::-1]))

    me.vertices.foreach_set("co", ve.ravel())
    me.polygons.foreach_set("loop_total", face_lengths)
    me.polygons.foreach_set("loop_start", loops)
    me.polygons.foreach_set("vertices", v_out)
    
    me.update(calc_edges=True)
    #me.validate(verbose=True)

    return me


def read_verts(mesh):
    mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
    mesh.vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(mesh.vertices), 3))   


def read_loops(mesh):
    loops = np.zeros((len(mesh.polygons)), dtype=np.int)
    mesh.polygons.foreach_get("loop_total", loops)
    return loops 

     
def read_edges(mesh):
    fastedges = np.zeros((len(mesh.edges)*2), dtype=np.int) # [0.0, 0.0] * len(mesh.edges)
    mesh.edges.foreach_get("vertices", fastedges)
    return np.reshape(fastedges, (len(mesh.edges), 2))


def read_norms(mesh):
    mverts_no = np.zeros((len(mesh.vertices)*3), dtype=np.float)
    mesh.vertices.foreach_get("normal", mverts_no)
    return np.reshape(mverts_no, (len(mesh.vertices), 3))


def read_polygon_verts(mesh):
    polys = np.zeros((len(mesh.polygons)*4), dtype=np.uint32)
    mesh.polygons.foreach_get("vertices", faces)
    return polys


def safe_bincount(data, weights, dts, conn):
    bc = np.bincount(data, weights)
    dts[:len(bc)] += bc
    bc = np.bincount(data)
    conn[:len(bc)] += bc
    return (dts, conn)


def calc_normals(fastverts, fastnorms, fastedges):
    """ Calculates curvature [0, 1] for specified mesh """
    edge_a, edge_b = fastedges[:,0], fastedges[:,1]
    
    tvec = fastverts[edge_b] - fastverts[edge_a]
    tvlen = np.linalg.norm(tvec, axis=1)    

    # normalize vectors
    tvec = (tvec.T / tvlen).T 

    # adjust the minimum of what is processed   
    edgelength = tvlen * 100
    edgelength = np.where(edgelength < 1, 1.0, edgelength)

    vecsums = np.zeros(fastverts.shape[0], dtype=np.float) 
    connections = np.zeros(fastverts.shape[0], dtype=np.float) 

    # calculate normal differences to the edge vector in the first edge vertex
    totdot = (np.einsum('ij,ij->i', tvec, fastnorms[edge_a]))/edgelength
    safe_bincount(edge_a, totdot, vecsums, connections)

    # calculate normal differences to the edge vector  in the second edge vertex
    totdot = (np.einsum('ij,ij->i', -tvec, fastnorms[edge_b]))/edgelength
    safe_bincount(edge_b, totdot, vecsums, connections)

    # (approximate gaussian) curvature is the average difference of 
    # edge vectors to surface normals (from dot procuct cosine equation)
    curve = 1.0 - np.arccos(vecsums/connections)/np.pi

    # 1 = max curvature, 0 = min curvature, 0.5 = zero curvature
    curve -= 0.5
    curve /= np.max([np.amax(curve), np.abs(np.amin(curve))])
    curve += 0.5
    return curve


def mesh_smooth_filter_variable(data, fastverts, fastedges, iterations):
    """ Smooths variables in data [-1, 1] over the mesh topology """
    # vert indices of edges
    edge_a, edge_b = fastedges[:,0], fastedges[:,1]
    tvlen = np.linalg.norm(fastverts[edge_b] - fastverts[edge_a], axis=1)
    edgelength = np.where(tvlen<1, 1.0, tvlen)

    data_sums = np.zeros(fastverts.shape[0], dtype=np.float) 
    connections = np.zeros(fastverts.shape[0], dtype=np.float) 

    # longer the edge distance to datapoint, less it has influence

    for _ in range(iterations):
        # step 1
        per_vert = data[edge_b]/edgelength
        safe_bincount(edge_a, per_vert, data_sums, connections)
        eb_smooth = data_sums/connections
        
        per_vert = eb_smooth[edge_a]/edgelength
        safe_bincount(edge_b, per_vert, data_sums, connections)

        new_data = data_sums/connections

        # step 2
        data_sums = np.zeros(data_sums.shape)
        connections = np.zeros(connections.shape)

        per_vert = data[edge_a]/edgelength
        safe_bincount(edge_b, per_vert, data_sums, connections)
        ea_smooth = data_sums/connections
        
        per_vert = ea_smooth[edge_b]/edgelength
        safe_bincount(edge_a, per_vert, data_sums, connections)

        new_data += data_sums/connections

        # limit between -1 and 1
        new_data /= np.max([np.amax(new_data), np.abs(np.amin(new_data))])
        data = new_data

    return new_data


def vdb_remesh(verts, tris, quads, iso, adapt, only_quads, vxsize, filter_iterations, filter_width, filter_style, grid=None):

    iso *= vxsize

    def _read(verts, tris, quads, vxsize):
        print("vdb: read voxels from mesh")
        vtransform = vdb.createLinearTransform(voxelSize=vxsize)

        if len(tris) == 0:
            grid = vdb.FloatGrid.createLevelSetFromPolygons(verts, quads=quads, transform=vtransform)
        elif len(quads) == 0:
            grid = vdb.FloatGrid.createLevelSetFromPolygons(verts, triangles=tris, transform=vtransform)
        else:
            grid = vdb.FloatGrid.createLevelSetFromPolygons(verts, tris, quads, transform=vtransform)

        bb = grid.evalActiveVoxelBoundingBox()
        bb_size = (bb[1][0]-bb[0][0], bb[1][1]-bb[0][1], bb[1][2]-bb[0][2])
        print("vdb_remesh: new grid {} voxels".format(bb_size))

        return grid
    
    saved_grid = None
    if grid == None:
        grid = _read(verts, tris, quads, vxsize)
    else:
        saved_grid = grid
        grid = grid.deepCopy()

    def _filter_numpy(iterations, gr, fname):
        print("vdb: blur")
        bb = gr.evalActiveVoxelBoundingBox()
        bb_size = (bb[1][0]-bb[0][0], bb[1][1]-bb[0][1], bb[1][2]-bb[0][2])
        print("Smoothing on {} bounding box".format(bb_size))
        if bb_size[0] * bb_size[1] * bb_size[2] < MAX_SMOOTHING:
            array = np.empty(bb_size, dtype=np.float)
            gr.copyToArray(array, ijk=bb[0])
            aavg = None
            for _ in range(iterations):

                if fname == "blur":
                    surrounding = (
                        np.roll(array, -1, axis=0) + np.roll(array, 1, axis=0) + \
                        np.roll(array, -1, axis=1) + np.roll(array, 1, axis=1) + \
                        np.roll(array, -1, axis=2) + np.roll(array, 1, axis=2))
                    array = (array + surrounding) / 7

                if fname == "sharpen":
                    surrounding = (
                        np.roll(array, -1, axis=0) + np.roll(array, 1, axis=0) + \
                        np.roll(array, -1, axis=1) + np.roll(array, 1, axis=1) + \
                        np.roll(array, -1, axis=2) + np.roll(array, 1, axis=2))
                    array = (array) * 0.5 + (array*7 - surrounding) * 0.5

                if fname == "edge":
                    surrounding = (
                        np.roll(array, -1, axis=0) + np.roll(array, 1, axis=0) + \
                        np.roll(array, -1, axis=1) + np.roll(array, 1, axis=1) + \
                        np.roll(array, -1, axis=2) + np.roll(array, 1, axis=2))
                    array = surrounding - 4*array

            gr.copyFromArray(array, ijk=bb[0])
        else:
            print("Smoothing bounding box exceeded maximum size. Skipping.")

    def _write(gr):
        print("vdb: write voxels to polygons")
        fit = filter_iterations if filter_iterations > 0 else 0
        verts, tris, quads = gr.convertToComplex(iso, adapt, fit, filter_width)

        return (verts, tris, quads)

    return (_write(grid), grid if saved_grid == None else saved_grid)

def read_mesh(mesh):
    print("vdb_remesh: read mesh")
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
    print("vdb_remesh: read bmesh")
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


def bmesh_update_normals(obj):
    # calc normals
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    #bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()

def project_to_nearest(source, target):
    print("vdb_remesh: project to nearest")
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

def project_to_nearest_numeric(source, target):
    print("vdb_remesh: project to nearest (numpy)")

    if len(source[0]) == 0:
        return

    res = target.closest_point_on_mesh(source[0][0].tolist())
    
    if len(res) > 3:
        # new style 2.79
        rnum = 1
    else:
        # old style 2.76
        rnum = 0

    vfunc = target.closest_point_on_mesh
    for i, vtx in enumerate(source[0]):
        source[0][i] = vfunc(vtx)[rnum]
    

class VDBRemeshOperator(bpy.types.Operator):
    """OpenVDB Remesh"""
    bl_idname = "object.vdbremesh_operator"
    bl_label = "OpenVDB remesh"
    bl_options = {'REGISTER', 'UNDO'}

    voxel_size_def = \
        bpy.props.EnumProperty(items=[('relative', 'Relative', 'Voxel size is defined in relation to the object size'), 
                                    ('absolute', 'Absolute', 'Voxel size is defined in world coordinates')],
            name="voxel_size_def", default='relative')

    voxel_size_world = bpy.props.FloatProperty(
            name="Voxel size",
            description="Voxel size defined in world coordinates",
            min=MIN_VOXEL_SIZE, max=MAX_VOXEL_SIZE,
            default=0.05)

    voxel_size_object = bpy.props.FloatProperty(
            name="Voxel size (relative)",
            description="Voxel size in relation to the objects longest bounding box edge",
            min=0.001, max=0.25,
            default=0.005)

    isovalue = bpy.props.FloatProperty(
            name="Isovalue",
            description="Isovalue",
            min=-3.0, max=3.0,
            default=0.0)

    adaptivity = bpy.props.FloatProperty(
            name="Adaptivity",
            description="Adaptivity",
            min=0.0, max=1.0,
            default=0.0001)

    filter_style = \
        bpy.props.EnumProperty(items=[('blur', 'Blur', 'Blur voxels'), 
                                    ('sharpen', 'Sharpen', 'Sharpen voxels'),
                                    ('edge', 'Edge', 'Edge enhance')],
            name="Filter style", default='blur')

    filter_iterations = bpy.props.IntProperty(
            name="Gaussian iterations",
            description="Gaussian iterations",
            min=0, max=10,
            default=0)

    filter_width = bpy.props.IntProperty(
            name="Gaussian width",
            description="Gaussian width",
            min=1, max=10,
            default=1)

    filter_quality = bpy.props.FloatProperty(
            name="Gaussian sharpness",
            description="Gaussian sharpness",
            min=2.0, max=10.0,
            default=5.0)

    only_quads = bpy.props.BoolProperty(
            name="Quads only",
            description="Construct the mesh using only quad topology",
            default=False)

    smooth = bpy.props.BoolProperty(
            name="Smooth",
            description="Smooth shading toggle",
            default=True)

    nearest = bpy.props.BoolProperty(
            name="Project to nearest",
            description="Project generated mesh points to nearest surface point",
            default=False)

    grid = None
    grid_voxelsize = None
    max_polys_reached = False

    @classmethod
    def poll(cls, context):
        return context.active_object is not None


    def execute(self, context):
        print("vdb_remesh: execute")
        if DEBUG:
            pr = cProfile.Profile()
            pr.enable()

        # voxel_def = context.scene.vdb_remesh_voxel_size_def
        # voxel_world = context.scene.vdb_remesh_voxel_size_world
        # voxel_object = context.scene.vdb_remesh_voxel_size_object

        voxel_size = 0.07

        if self.voxel_size_def == "relative":
            voxel_size = max(context.active_object.dimensions) * self.voxel_size_object
        else:
            voxel_size = self.voxel_size_world
        

        # apply modifiers for the active object before remeshing
        for mod in context.active_object.modifiers:
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except RuntimeError as ex:
                print(ex)     

        # start remesh
        me = context.active_object.data

        if self.grid == None or self.grid_voxelsize != voxel_size:
            # caching
            self.grid = None
            
            bm = bmesh.new()
            bm.from_mesh(me)

            loops = read_loops(me)
            if np.max(loops) > 4:
                print("Mesh has ngons! Triangulating...")
                bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)

            self.grid_voxelsize = voxel_size
            startmesh = read_bmesh(bm)
            self.vert_0 = len(startmesh[0])
            bm.free()
        else:
            startmesh = (None, None, None)

        #vdb_remesh(verts, tris, quads, iso, adapt, only_quads, vxsize, filter_iterations, filter_width, filter_style, grid=None):
        
        new_mesh, self.grid = vdb_remesh(startmesh[0], startmesh[1], startmesh[2], self.isovalue, \
            self.adaptivity, self.only_quads, voxel_size, self.filter_iterations, self.filter_width, \
            self.filter_style, grid=self.grid)

        print("vdb_remesh: new mesh {}".format([i.shape for i in new_mesh]))
        self.vert_1 = len(new_mesh[0])
        self.face_1 = len(new_mesh[1])+len(new_mesh[2])

        if self.face_1 < MAX_POLYGONS:
            self.max_polys_reached = False

            #if self.project_nearest:
                #context.scene.update()
                #project_to_nearest_numeric(new_mesh, context.active_object)
   
            remeshed = write_fast(*new_mesh)

            context.active_object.data = remeshed

            if self.nearest:
                def _project_wrap():
                    temp_object = bpy.data.objects.new("temp.remesher.947", me)
                    temp_object.matrix_world = context.active_object.matrix_world

                    
                    bpy.ops.object.modifier_add(type='SHRINKWRAP')
                    context.object.modifiers["Shrinkwrap"].target = temp_object
                    
                    for mod in context.active_object.modifiers:
                        try:
                            bpy.ops.object.modifier_apply(modifier=mod.name)
                        except RuntimeError as ex:
                            print(ex)    

                    objs = bpy.data.objects
                    objs.remove(objs["temp.remesher.947"], True)
                
                _project_wrap()


            # calc normals
            #bmesh_update_normals(context.active_object)

            def _make_normals_consistent():
                bpy.ops.object.mode_set(mode = 'EDIT')
                bpy.ops.mesh.normals_make_consistent()
                bpy.ops.object.mode_set(mode = 'OBJECT')

            #_make_normals_consistent()

            # flip normals
            """
            obj = context.active_object
            norms = np.zeros((len(obj.data.polygons)*3), dtype=np.float)
            obj.data.polygons.foreach_get("normal", norms)
            norms = -norms
            obj.data.polygons.foreach_set("normal", norms)
            
            norms = np.zeros((len(obj.data.vertices)*3), dtype=np.float)
            obj.data.vertices.foreach_get("normal", norms)
            norms = -norms
            obj.data.vertices.foreach_set("normal", norms)
            """

            if self.only_quads:   
                def _decimate_flat_slow():
                    #obj.data.update()
                    print("vdb_remesh: decimate flat: calculating curvature...")

                    mesh = context.active_object.data
                    fastverts = read_verts(mesh)
                    fastedges = read_edges(mesh)
                    fastnorms = read_norms(mesh) 

                    values = calc_normals(fastverts, fastnorms, fastedges)

                    values = (values-0.5)*2.0
                    values = mesh_smooth_filter_variable(values, fastverts, fastedges, 5)

                    bm = bmesh.new()
                    bm.from_mesh(context.active_object.data)
                    bm.verts.ensure_lookup_table()
                    c_verts = []
                    for i, v in enumerate(bm.verts):
                        if abs(values[i]) < 0.01:
                            c_verts.append(v)

                    print("vdb_remesh: decimate flat: collapsing verts (bmesh.ops)")
                    bmesh.ops.dissolve_verts(bm, verts=c_verts) #, use_face_split, use_boundary_tear)

                    bm.to_mesh(context.active_object.data)
                    bm.free()



            if self.smooth:
                bpy.ops.object.shade_smooth()

        else:
            self.max_polys_reached = True

        if DEBUG:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s)
            ps.strip_dirs().sort_stats(sortby).print_stats()
            print(s.getvalue())

        print("vdb_remesh: exit")

        return {'FINISHED'}

    
    def draw(self, context):
        layout = self.layout
        col = layout.column()

        if self.max_polys_reached:
            row = col.row()
            row.label(text="Max poly count reached (>{})".format(MAX_POLYGONS))
            row = col.row()
            row.label(text="Skipping writing to mesh.")

        row = layout.row()
        row.prop(self, "voxel_size_def", expand=True, text="Island margin quality/performance")

        row = layout.row()
        col = row.column(align=True)
        if self.voxel_size_def == "relative":
            col.prop(self, "voxel_size_object")
        else:
            col.prop(self, "voxel_size_world")

        row = layout.row()
        col = row.column(align=True)
        col.prop(self, "isovalue")

        col.prop(self, "adaptivity")

        #row = layout.row()
        #row.prop(self, "filter_style", expand=True, text="Type of filter to be iterated on the voxels")
        #row = layout.row()
        row = layout.row()
        col = row.column(align=True)

        col.prop(self, "filter_iterations")
        col.prop(self, "filter_width")
        col.prop(self, "filter_quality")

        row = layout.row()
        #row.prop(self, "only_quads")
        row.prop(self, "smooth")
        row.prop(self, "nearest")

        if hasattr(self, 'vert_0'):
            infotext = "Change: {:.2%}".format(self.vert_1/self.vert_0)
            row = layout.row()
            row.label(text=infotext)
            row = layout.row()
            row.label(text="Verts: {}, Polys: {}".format(self.vert_1, self.face_1))

            row = layout.row()
            row.label(text="Cache: {} voxels".format(self.grid.activeVoxelCount()))


        

class VDBRemeshPanel(bpy.types.Panel):
    """OpenVDB remesh operator panel"""
    bl_label = "VDB remesh"
    bl_idname = "object.vdbremesh_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_category = "Retopology"

    def draw(self, context):
        layout = self.layout

        """
        row = layout.row()
        row.prop(context.scene, "vdb_remesh_voxel_size_def", expand=True, text="Island margin quality/performance")

        row = layout.row()
        col = row.column(align=True)
        if context.scene.vdb_remesh_voxel_size_def == "relative":
            col.prop(context.scene, "vdb_remesh_voxel_size_object")
        else:
            col.prop(context.scene, "vdb_remesh_voxel_size_world")

        col.prop(context.scene, "vdb_remesh_isovalue")
        col.prop(context.scene, "vdb_remesh_adaptivity")
        col.prop(context.scene, "vdb_remesh_blur")

        row = layout.row()
        row.prop(context.scene, "vdb_remesh_only_quads")
        row.prop(context.scene, "vdb_remesh_smooth")
        row.prop(context.scene, "vdb_remesh_project_nearest")
        """

        row = layout.row()
        row.scale_y = 2.0
        row.operator(VDBRemeshOperator.bl_idname, text="OpenVDB remesh")


def register():
    bpy.utils.register_class(VDBRemeshOperator)
    bpy.utils.register_class(VDBRemeshPanel)

    """
    save_to = bpy.types.Scene

    save_to.vdb_remesh_voxel_size_def = \
        bpy.props.EnumProperty(items=[('relative', 'Relative', 'Voxel size is defined in relation to the object size'), 
                                    ('absolute', 'Absolute', 'Voxel size is defined in world coordinates')],
            name="s_packing_marginquality", default='relative')

    save_to.vdb_remesh_voxel_size_world = bpy.props.FloatProperty(
            name="Voxel size",
            description="Voxel size defined in world coordinates",
            min=MIN_VOXEL_SIZE, max=MAX_VOXEL_SIZE,
            default=0.05)

    save_to.vdb_remesh_voxel_size_object = bpy.props.FloatProperty(
            name="Voxel size (relative)",
            description="Voxel size in relation to the objects longest bounding box edge",
            min=0.001, max=0.25,
            default=0.01)

    save_to.vdb_remesh_isovalue = bpy.props.FloatProperty(
            name="Isovalue",
            description="Isovalue",
            min=-3.0, max=3.0,
            default=0.0)

    save_to.vdb_remesh_adaptivity = bpy.props.FloatProperty(
            name="Adaptivity",
            description="Adaptivity",
            min=0.0, max=1.0,
            default=0.01)

    save_to.vdb_remesh_blur = bpy.props.IntProperty(
            name="Blur",
            description="Blur iterations",
            min=0, max=10,
            default=0)

    save_to.vdb_remesh_only_quads = bpy.props.BoolProperty(
            name="Quads only",
            description="Construct the mesh using only quad topology",
            default=False)

    save_to.vdb_remesh_smooth = bpy.props.BoolProperty(
            name="Smooth",
            description="Smooth shading toggle",
            default=True)

    save_to.vdb_remesh_project_nearest = bpy.props.BoolProperty(
            name="Project to nearest",
            description="Project generated mesh points to nearest surface point",
            default=False)
    """


def unregister():
    bpy.utils.unregister_class(VDBRemeshOperator)
    bpy.utils.unregister_class(VDBRemeshPanel)


if __name__ == "__main__":
    register()
    #bpy.ops.object.vdbremesh_operator()


