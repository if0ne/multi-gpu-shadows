use std::{
    cell::RefCell,
    collections::HashMap,
    path::{Path, PathBuf},
    rc::{Rc, Weak},
};

use oxidx::dx;

use crate::rhi::{self, FRAMES_IN_FLIGHT};

#[derive(Clone, Copy, Debug)]
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
        let (gltf, buffers, images) = gltf::import(&path).expect("Failed to open gltf model");

        let cmd_buffer = rhi::CommandBuffer::new(device, dx::CommandListType::Direct, false);
        cmd_buffer.begin(device, false);

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

        cmd_buffer.end();
        queue.submit(&[&cmd_buffer]);
        let value = queue.signal();
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
        let transform = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        transform
    }
}
