import bpy
import numpy
import math

print("bevelshader: hello")

class BevelShader:
    def init_images(self, context):
        self.input_image = context.scene.seamless_input_image
        self.target_image = context.scene.seamless_generated_name

        print("Creating images...")
        self.size = bpy.data.images[self.input_image].size
        self.xs = self.size[0]
        self.ys = self.size[1]

        # copy image data into much more performant numpy arrays
        self.pixels = numpy.array(bpy.data.images[self.input_image].pixels)
        self.pixels = self.pixels.reshape((self.ys, self.xs, 4))

        # if target image exists, change the size to fit
        if self.target_image in bpy.data.images:
            bpy.data.images[self.target_image].scale(self.xs, self.ys)
            self.image = bpy.data.images[self.target_image]
        else:
            self.image = bpy.data.images.new(self.target_image, width=self.xs, height=self.ys)

        self.pixels[:, :, 3] = 1.0  # alpha is always 1.0 everywhere

    def finish_images(self):
        self.image.pixels = self.pixels.flatten()
        #bpy.ops.image.invert(invert_r=False, invert_g=False, invert_b=False, invert_a=False)

    def fast_gaussian(self, s):
        d = int(2 ** s)
        tpx = self.pixels
        ystep = tpx.shape[1]
        while d > 1:
            tpx = (tpx * 2.0 + numpy.roll(tpx, -d * 4) + numpy.roll(tpx, d * 4)) / 4.0
            tpx = (tpx * 2.0 + numpy.roll(tpx, -d * (ystep * 4)) + numpy.roll(tpx, d * (ystep * 4))) / 4.0
            d = int(d / 2.0)
        self.pixels = tpx

    def convolution(self, intens, sfil):
        # source, intensity, convolution matrix
        ssp = self.pixels
        tpx = numpy.zeros(ssp.shape, dtype=float)
        tpx[:, :, 3] = 1.0
        ystep = int(4 * ssp.shape[1])
        norms = 0
        for y in range(sfil.shape[0]):
            for x in range(sfil.shape[1]):
                tpx += numpy.roll(ssp, (x - int(sfil.shape[1] / 2)) * 4 + (y - int(sfil.shape[0] / 2)) * ystep) * sfil[y, x]
                norms += sfil[y, x]
        if norms > 0:
            tpx /= norms
        return ssp + (tpx - ssp) * intens
        
    def blur(self, s, intensity):
        self.pixels = self.convolution(intensity, numpy.ones((1 + s * 2, 1 + s * 2), dtype=float))
        
    def inside_tri(self, pt, v1, v2, v3):
        def signn(p1, p2, p3):
            return (p1[0]-p3[0]) * (p2[1]-p3[1]) - (p2[0]-p3[0]) * (p1[1]-p3[1])

        b1 = signn(pt, v1, v2) < 0.0
        b2 = signn(pt, v2, v3) < 0.0
        b3 = signn(pt, v3, v1) < 0.0

        return (b1 == b2) and (b2 == b3)
    
    def draw_point(self, tpx, xloc, yloc, cval, tri):
        #if self.inside_tri((xloc, yloc), tri[0], tri[1], tri[2]):
        tpx[yloc % self.ys, xloc % self.xs,:] = cval    
        
    def draw_bevels(self):
        
        # ov.data.use_auto_smooth
        # ob.data.edges[0].use_edge_sharp
                
        print("draw bevels")
        tpx = self.pixels
        
        ob = bpy.context.object
        uv_layer = ob.data.uv_layers.active.data
        
        # map edge to index
        edgemap = {}
        for i, edge in enumerate(ob.data.edges):
            edgemap[(edge.vertices[0], edge.vertices[1])] = i #edge.use_edge_sharp        
        
        # list of faces connected by an edge
        edgesharp = {}
        connecting_edges = {}
        for polyid, poly in enumerate(ob.data.polygons):
            for edge in poly.edge_keys:
                if not edge in connecting_edges:
                    connecting_edges[edge] = [polyid]
                else:
                    connecting_edges[edge].append(polyid)
                    
                getedge = ob.data.edges[edgemap[edge] if edge in edgemap else edgemap[(edge[1], edge[0])]]
                edgesharp[edge] = getedge.use_edge_sharp

        step = 1.0/float(self.xs)
        for polyid, poly in enumerate(ob.data.polygons):
            uvs = [uv_layer[i].uv for i in poly.loop_indices]
            tri_center = (sum(i[0] for i in uvs)/len(uvs), sum(i[1] for i in uvs)/len(uvs))
            for i, uv in enumerate(uvs):
                if edgesharp[poly.edge_keys[i]]: #poly.edge_keys[i] in edgesharp and edgesharp[poly.edge_keys[i]]:                     
                    a = uv
                    b = uvs[(i+1)%len(uvs)]
                    
                    new_tri = numpy.array([a, b, tri_center])
                    new_tri[:,0] *= self.xs
                    new_tri[:,1] *= self.ys
                    
                    xdif = b[0]-a[0]
                    ydif = b[1]-a[1]
                    
                    ablen = math.sqrt(xdif**2.0 + ydif**2.0)
                    
                    # generate edge normal
                    faces = connecting_edges[poly.edge_keys[i]]
                    cval = numpy.zeros((4), dtype=numpy.float16)
                    for f in faces:
                        cval[:3] += numpy.array(ob.data.polygons[f].normal)
                    cval[:3] /= numpy.sqrt(cval[0]**2 + cval[1]**2 + cval[2]**2)     
                    cval[:3] = (cval[:3]+1.0)/2.0
                    cval[3] = 1.0
                    
                    for l in numpy.arange(0, ablen, step):
                        x = a[0] + xdif * float(l) / ablen
                        y = a[1] + ydif * float(l) / ablen 
                        xloc = int(x*float(self.xs))
                        yloc = int(y*float(self.ys))

                        self.draw_point(tpx, xloc  , yloc  , cval, new_tri)
                        self.draw_point(tpx, xloc+1, yloc  , cval, new_tri)
                        self.draw_point(tpx, xloc-1, yloc  , cval, new_tri)
                        self.draw_point(tpx, xloc  , yloc+1, cval, new_tri)
                        self.draw_point(tpx, xloc  , yloc-1, cval, new_tri)
        
        self.pixels = tpx
                    

bshader = BevelShader()
bshader.init_images(bpy.context)
bshader.draw_bevels()
bshader.blur(3, 1.0)
bshader.draw_bevels()
bshader.blur(2, 1.0)
bshader.finish_images()

print("bevelshader: goodbye")