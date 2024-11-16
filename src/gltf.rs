use std::{
    cell::RefCell,
    collections::HashMap,
    path::{Path, PathBuf},
    rc::{Rc, Weak},
};

use gltf::Gltf;
use oxidx::dx;

use crate::rhi::{self, FRAMES_IN_FLIGHT};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    pub pos: glam::Vec3,
    pub uv: glam::Vec2,
    pub normals: glam::Vec3,
    pub tangent: glam::Vec3,
}

pub struct Material {
    pub albedo: Option<rhi::Texture>,
}

pub struct Primitive {
    pub vertex_buffer: rhi::Buffer,
    pub index_buffer: rhi::Buffer,

    pub vtx_count: u32,
    pub idx_count: u32,
    pub mat_index: Option<usize>,
}

pub struct Node {
    pub primitives: Vec<Primitive>,
    pub model_buffer: [rhi::Buffer; FRAMES_IN_FLIGHT],

    pub name: String,
    pub transform: glam::Mat4,
    pub parent: Option<Weak<RefCell<Node>>>,
    pub children: Vec<Rc<RefCell<Node>>>,
}

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
        tracker: &rhi::GpuResourceTracker,
        buffers: &[gltf::buffer::Data]
    ) {
        let transform = Self::compute_transform(node);
        let new_node = Rc::new(RefCell::new(Node {
            primitives: vec![],
            model_buffer: std::array::from_fn(|_| {
                rhi::Buffer::new(
                    tracker,
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
                Self::process_primitive(tracker, &primitive, &mut new_node.borrow_mut(), buffers);
            }
        }

        for child in node.children() {
            Self::process_node(&child, new_node.clone(), tracker, buffers);
        }
    }

    fn process_primitive(
        tracker: &rhi::GpuResourceTracker,
        primitive: &gltf::Primitive,
        node: &mut Node,
        buffers: &[gltf::buffer::Data]
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

        let Some(tangets) = reader.read_tangents() else {
            return;
        };

        let vertices = positions
            .zip(uvs.into_f32())
            .zip(normals)
            .zip(tangets)
            .map(|(((pos, uv), normals), tangets)| {
                let [x, y, z, ..] = tangets;
                Vertex {
                    pos: glam::Vec3::from_array(pos),
                    uv: glam::Vec2::from_array(uv),
                    normals: glam::Vec3::from_array(normals),
                    tangent: glam::Vec3::from_array([x, y, z]),
                }
            })
            .collect::<Vec<_>>();

        let vertex_count = vertices.len();
        let index_count = primitive.indices().unwrap().count();

        let vertex_buffer = rhi::Buffer::new(
            tracker,
            vertex_count * std::mem::size_of::<Vertex>(),
            std::mem::size_of::<Vertex>(),
            rhi::BufferType::Vertex,
            false,
            format!("{} Vertex Buffer", node.name),
        );

        let index_buffer = rhi::Buffer::new(
            tracker,
            index_count * std::mem::size_of::<u32>(),
            std::mem::size_of::<u32>(),
            rhi::BufferType::Index,
            false,
            format!("{} Index Buffer", node.name),
        );

        node.primitives.push(Primitive {
            vertex_buffer,
            index_buffer,
            vtx_count: vertex_count as u32,
            idx_count: index_count as u32,
            mat_index: None,
        });
    }

    pub fn load(
        device: &rhi::Device,
        tracker: &rhi::GpuResourceTracker,
        queue: &rhi::CommandQueue,
        fence: &rhi::Fence,
        path: impl AsRef<Path>,
    ) -> Self {
        let (gltf, buffers, images) = gltf::import(&path).expect("Failed to open gltf model");

        let cmd_buffer = rhi::CommandBuffer::new(device, dx::CommandListType::Direct, false);
        cmd_buffer.begin(device, false);

        let scene = gltf.scenes().next().expect("No gltf scenes");

        let root = Rc::new(RefCell::new(Node {
            primitives: vec![],
            model_buffer: std::array::from_fn(|_| {
                rhi::Buffer::new(
                    tracker,
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
            Self::process_node(&node, root.clone(), tracker, &buffers);
        }

        cmd_buffer.end();
        queue.submit(&[&cmd_buffer]);
        let value = queue.signal(fence);
        fence.wait(value);

        Self {
            path: path.as_ref().to_path_buf(),
            directory: path.as_ref().parent().unwrap().to_path_buf(),
            root,
            materials: vec![],
            textures: HashMap::new(),
        }
    }

    fn compute_transform(node: &gltf::Node) -> glam::Mat4 {
        let transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        transform
    }
}
