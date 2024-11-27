use std::path::Path;

use glam::{Mat4, Vec3, Vec4};
use gltf::image;

#[derive(Clone, Debug)]
pub enum ImageSource {
    Path(std::path::PathBuf),
    Data(Vec<u8>),
}

#[derive(Clone, Debug)]
pub enum MaterialSlot {
    Placeholder([f32; 4]),
    Image(usize),
}

#[derive(Clone, Debug)]
pub struct Material {
    pub diffuse: MaterialSlot,
    pub normal: MaterialSlot,
}

#[derive(Clone, Default)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub indices: Vec<u32>,

    pub sub_meshes: Vec<Submesh>,
    pub materials: Vec<Material>,
    pub images: Vec<ImageSource>,
}

#[derive(Clone, Copy, Debug)]
pub struct Submesh {
    pub index_count: u32,
    pub start_index_location: u32,
    pub base_vertex_location: u32,
    pub material_idx: usize,
}

fn iter_gltf_node_tree<F: FnMut(&gltf::scene::Node, Mat4)>(
    node: &gltf::scene::Node,
    xform: Mat4,
    f: &mut F,
) {
    let node_xform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let xform = xform * node_xform;

    f(node, xform);
    for child in node.children() {
        iter_gltf_node_tree(&child, xform, f);
    }
}

impl Mesh {
    pub fn load(path: impl AsRef<Path>) -> Self {
        let (gltf, buffers, images) = gltf::import(path).expect("Failed to open file");

        let scene = gltf
            .default_scene()
            .or_else(|| gltf.scenes().next())
            .expect("Failed to fetch scene");
        let mut res = Mesh::default();

        for image in gltf.images() {
            match image.source() {
                image::Source::Uri { uri, .. } => {
                    //todo!()
                }
                image::Source::View { view, .. } => {
                    //todo!()
                }
            }
        }

        res.materials = gltf
            .materials()
            .map(|m| Material {
                diffuse: MaterialSlot::Placeholder(m.pbr_metallic_roughness().base_color_factor()),
                normal: MaterialSlot::Placeholder([0.0, 0.0, 0.0, 0.0]),
            })
            .collect();

        let mut process_node = |node: &gltf::scene::Node, xform: Mat4| {
            if let Some(mesh) = node.mesh() {
                let flip_winding_order = xform.determinant() < 0.0;

                for prim in mesh.primitives() {
                    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                    let positions = if let Some(iter) = reader.read_positions() {
                        iter.collect::<Vec<_>>()
                    } else {
                        return;
                    };

                    let normals = if let Some(iter) = reader.read_normals() {
                        iter.collect::<Vec<_>>()
                    } else {
                        return;
                    };

                    let (mut tangents, tangents_found) = if let Some(iter) = reader.read_tangents()
                    {
                        (iter.collect::<Vec<_>>(), true)
                    } else {
                        (vec![[1.0, 0.0, 0.0, 0.0]; positions.len()], false)
                    };

                    let (mut uvs, uvs_found) = if let Some(iter) = reader.read_tex_coords(0) {
                        (iter.into_f32().collect::<Vec<_>>(), true)
                    } else {
                        (vec![[0.0, 0.0]; positions.len()], false)
                    };

                    let mut colors = if let Some(iter) = reader.read_colors(0) {
                        iter.into_rgba_f32().collect::<Vec<_>>()
                    } else {
                        vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
                    };

                    let mut indices: Vec<u32>;
                    {
                        if let Some(indices_reader) = reader.read_indices() {
                            indices = indices_reader.into_u32().collect();
                        } else {
                            if positions.is_empty() {
                                return;
                            }

                            match prim.mode() {
                                gltf::mesh::Mode::Triangles => {
                                    indices = (0..positions.len() as u32).collect();
                                }
                                _ => {
                                    panic!("Primitive mode {:?} not supported yet", prim.mode());
                                }
                            }
                        }

                        if flip_winding_order {
                            for tri in indices.chunks_exact_mut(3) {
                                tri.swap(0, 2);
                            }
                        }
                    }

                    if !tangents_found && uvs_found {
                        mikktspace::generate_tangents(&mut TangentCalcContext {
                            indices: indices.as_slice(),
                            positions: positions.as_slice(),
                            normals: normals.as_slice(),
                            uvs: uvs.as_slice(),
                            tangents: tangents.as_mut_slice(),
                        });
                    }

                    let submesh = Submesh {
                        index_count: indices.len() as u32,
                        start_index_location: res.indices.len() as u32,
                        base_vertex_location: res.positions.len() as u32,
                        material_idx: prim.material().index().unwrap_or(0),
                    };

                    res.indices.append(&mut indices);
                    res.colors.append(&mut colors);

                    for v in positions {
                        let pos = (xform * Vec3::from(v).extend(1.0)).truncate();
                        res.positions.push(pos.into());
                    }

                    for v in normals {
                        let norm = (xform * Vec3::from(v).extend(0.0)).truncate().normalize();
                        res.normals.push(norm.into());
                    }

                    for v in tangents {
                        let v = Vec4::from(v);
                        let t = (xform * v.truncate().extend(0.0)).truncate().normalize();
                        res.tangents.push(
                            t.extend(v.w * if flip_winding_order { -1.0 } else { 1.0 })
                                .into(),
                        );
                    }

                    res.uvs.append(&mut uvs);

                    res.sub_meshes.push(submesh);
                }
            }
        };

        let xform = Mat4::IDENTITY;
        for node in scene.nodes() {
            iter_gltf_node_tree(&node, xform, &mut process_node);
        }

        res
    }
}

struct TangentCalcContext<'a> {
    indices: &'a [u32],
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    uvs: &'a [[f32; 2]],
    tangents: &'a mut [[f32; 4]],
}

impl<'a> mikktspace::Geometry for TangentCalcContext<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.indices[face * 3 + vert] as usize]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.indices[face * 3 + vert] as usize]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[self.indices[face * 3 + vert] as usize] = tangent;
    }
}

/*#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    pub pos: glam::Vec3,
    pub normals: glam::Vec3,
    pub uv: glam::Vec2,
}

#[derive(Debug)]
pub struct Material {
    pub albedo: Option<rhi::Texture>,
}

#[derive(Debug)]
pub struct Primitive {
    pub vertex_buffer: rhi::Buffer,
    pub index_buffer: rhi::Buffer,

    pub vtx_count: u32,
    pub idx_count: u32,
    pub mat_index: Option<usize>,
}

#[derive(Debug)]
pub struct Node {
    pub primitives: Vec<Primitive>,
    pub model_buffer: [rhi::Buffer; FRAMES_IN_FLIGHT],

    pub name: String,
    pub transform: glam::Mat4,
    pub parent: Option<Weak<RefCell<Node>>>,
    pub children: Vec<Rc<RefCell<Node>>>,
}

#[derive(Debug)]
pub struct Model {
    pub path: PathBuf,
    pub directory: PathBuf,

    pub root: Rc<RefCell<Node>>,
    pub materials: Vec<Material>,
    pub textures: HashMap<String, rhi::Texture>,
}

impl Model {
    fn process_node(
        node: &gltf::Node,
        parent: Rc<RefCell<Node>>,
        device: &rhi::Device,
        buffers: &[gltf::buffer::Data],
        staging_buffers: &mut Vec<rhi::Buffer>,
        cmd_buffer: &rhi::CommandBuffer,
    ) {
        let transform = Self::compute_transform(node);
        let new_node = Rc::new(RefCell::new(Node {
            primitives: vec![],
            model_buffer: std::array::from_fn(|_| {
                rhi::Buffer::new(
                    device,
                    512,
                    0,
                    rhi::BufferType::Constant,
                    false,
                    "CBV".to_string(),
                )
            }),
            name: node.name().unwrap_or("Unnamed Node").to_string(),
            transform,
            parent: Some(Rc::downgrade(&parent)),
            children: vec![],
        }));

        parent.borrow_mut().children.push(new_node.clone());

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                Self::process_primitive(
                    device,
                    &primitive,
                    &mut new_node.borrow_mut(),
                    buffers,
                    staging_buffers,
                    cmd_buffer,
                );
            }
        }

        for child in node.children() {
            Self::process_node(
                &child,
                new_node.clone(),
                device,
                buffers,
                staging_buffers,
                cmd_buffer,
            );
        }
    }

    fn process_primitive(
        device: &rhi::Device,
        primitive: &gltf::Primitive,
        node: &mut Node,
        buffers: &[gltf::buffer::Data],
        staging_buffers: &mut Vec<rhi::Buffer>,
        cmd_buffer: &rhi::CommandBuffer,
    ) {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            return;
        }

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        let Some(positions) = reader.read_positions() else {
            return;
        };

        let Some(uvs) = reader.read_tex_coords(0) else {
            return;
        };

        let Some(normals) = reader.read_normals() else {
            return;
        };

        let Some(indecies) = reader.read_indices() else {
            return;
        };

        let vertices = positions
            .zip(uvs.into_f32())
            .zip(normals)
            .map(|((pos, uv), normals)| Vertex {
                pos: glam::Vec3::from_array(pos),
                uv: glam::Vec2::from_array(uv),
                normals: glam::Vec3::from_array(normals),
            })
            .collect::<Vec<_>>();

        let vertex_count = vertices.len();
        let indecies = indecies.into_u32().collect::<Vec<_>>();

        let index_count = indecies.len();

        let mut vertex_staging = rhi::Buffer::new(
            device,
            vertex_count * std::mem::size_of::<Vertex>(),
            std::mem::size_of::<Vertex>(),
            rhi::BufferType::Copy,
            false,
            format!("{} Vertex Buffer", node.name),
        );

        {
            let map = vertex_staging.map::<Vertex>(None);
            map.pointer.clone_from_slice(&vertices);
        }

        let mut index_staging = rhi::Buffer::new(
            device,
            index_count * std::mem::size_of::<u32>(),
            std::mem::size_of::<u32>(),
            rhi::BufferType::Copy,
            false,
            format!("{} Index Buffer", node.name),
        );

        {
            let map = index_staging.map::<u32>(None);
            map.pointer.clone_from_slice(&indecies);
        }

        let vertex_buffer = rhi::Buffer::new(
            device,
            vertex_count * std::mem::size_of::<Vertex>(),
            std::mem::size_of::<Vertex>(),
            rhi::BufferType::Vertex,
            false,
            format!("{} Vertex Buffer", node.name),
        );

        let index_buffer = rhi::Buffer::new(
            device,
            index_count * std::mem::size_of::<u32>(),
            std::mem::size_of::<u32>(),
            rhi::BufferType::Index,
            false,
            format!("{} Index Buffer", node.name),
        );

        cmd_buffer.copy_buffer_to_buffer(&vertex_buffer, &vertex_staging);
        cmd_buffer.copy_buffer_to_buffer(&index_buffer, &index_staging);

        staging_buffers.push(vertex_staging);
        staging_buffers.push(index_staging);

        node.primitives.push(Primitive {
            vertex_buffer,
            index_buffer,
            vtx_count: vertex_count as u32,
            idx_count: index_count as u32,
            mat_index: None,
        });
    }

    pub fn load(device: &rhi::Device, queue: &rhi::CommandQueue, path: impl AsRef<Path>) -> Self {
        let (gltf, buffers, _) = gltf::import(&path).expect("Failed to open gltf model");

        let cmd_buffer = queue.get_command_buffer(device);
        cmd_buffer.begin(device);

        let scene = gltf.scenes().next().expect("No gltf scenes");

        let mut staging_buffers = vec![];

        let root = Rc::new(RefCell::new(Node {
            primitives: vec![],
            model_buffer: std::array::from_fn(|_| {
                rhi::Buffer::new(
                    device,
                    512,
                    0,
                    rhi::BufferType::Constant,
                    false,
                    "CBV".to_string(),
                )
            }),
            name: "root".to_string(),
            transform: glam::Mat4::IDENTITY,
            parent: None,
            children: vec![],
        }));

        for node in scene.nodes() {
            Self::process_node(
                &node,
                root.clone(),
                device,
                &buffers,
                &mut staging_buffers,
                &cmd_buffer,
            );
        }

        queue.push_cmd_buffer(cmd_buffer);
        let value = queue.execute();
        queue.wait_on_cpu(value);

        Self {
            path: path.as_ref().to_path_buf(),
            directory: path.as_ref().parent().unwrap().to_path_buf(),
            root,
            materials: vec![],
            textures: HashMap::new(),
        }
    }

    fn compute_transform(node: &gltf::Node) -> glam::Mat4 {
        glam::Mat4::from_cols_array_2d(&node.transform().matrix())
    }
}
*/
