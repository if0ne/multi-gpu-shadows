use crate::{
    rhi::{self, Device, DeviceMask, FRAMES_IN_FLIGHT},
    GpuMaterial, GpuTransform,
};
use std::{path::Path, sync::Arc};

use glam::{Mat4, Vec3, Vec4};
use oxidx::dx;

#[derive(Clone, Debug)]
pub enum ImageSource {
    Path(std::path::PathBuf),
    Data(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct Material {
    pub diffuse_color: [f32; 4],
    pub fresnel_r0: f32,
    pub roughness: f32,
    pub diffuse_map: Option<usize>,
    pub normal_map: Option<usize>,
}

#[derive(Clone, Default, Debug)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
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
        let (gltf, buffers, _) = gltf::import(&path).expect("Failed to open file");

        let scene = gltf
            .default_scene()
            .or_else(|| gltf.scenes().next())
            .expect("Failed to fetch scene");
        let mut res = Mesh::default();

        res.images = gltf
            .images()
            .map(|image| match image.source() {
                gltf::image::Source::View { view, .. } => {
                    let buffer_data = &buffers[view.buffer().index()];
                    let start = view.offset();
                    let end = start + view.length();
                    ImageSource::Data(buffer_data[start..end].to_vec())
                }
                gltf::image::Source::Uri { uri, .. } => {
                    let path = path
                        .as_ref()
                        .parent()
                        .unwrap_or_else(|| Path::new("./"))
                        .join(uri);
                    ImageSource::Path(path)
                }
            })
            .collect();

        res.materials = gltf
            .materials()
            .map(|m| Material {
                diffuse_color: m.pbr_metallic_roughness().base_color_factor(),
                fresnel_r0: m.pbr_metallic_roughness().metallic_factor(),
                roughness: m.pbr_metallic_roughness().roughness_factor(),
                diffuse_map: m
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .map(|t| t.texture().index()),
                normal_map: m.normal_texture().map(|m| m.texture().index()),
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

#[derive(Clone, Debug)]
pub struct GpuMeshBuilder<'a> {
    pub mesh: Mesh,
    pub devices: &'a [&'a Arc<Device>],
    pub normal_vb: DeviceMask,
    pub uv_vb: DeviceMask,
    pub tangent_vb: DeviceMask,
    pub materials: DeviceMask,
}

#[derive(Debug)]
pub struct GpuMesh {
    pub pos_vb: rhi::Buffer,
    pub normal_vb: rhi::Buffer,
    pub uv_vb: rhi::Buffer,
    pub tangent_vb: rhi::Buffer,

    pub ib: rhi::Buffer,
    pub gpu_materials: rhi::Buffer,
    pub transform: rhi::Buffer,

    pub images: Vec<rhi::Texture>,
    pub image_views: Vec<rhi::TextureView>,

    pub sub_meshes: Vec<Submesh>,
    pub materials: Vec<Material>,
}

impl GpuMesh {
    pub fn new(builder: GpuMeshBuilder<'_>) -> Self {
        let all_executors = builder
            .devices
            .iter()
            .map(|d| {
                let queue = &d.gfx_queue;
                let cmd_buffer = queue.get_command_buffer(d);
                cmd_buffer.begin(&d);

                (*d, queue, cmd_buffer)
            })
            .collect::<Vec<_>>();

        let (pos_vb, _pos_vb_staging) = {
            let mut pos_vb_staging = rhi::Buffer::copy::<[f32; 3]>(
                builder.mesh.positions.len(),
                "Vertex Position",
                builder.devices,
            );
            pos_vb_staging.write_all(&builder.mesh.positions);

            let pos_vb = rhi::Buffer::vertex::<[f32; 3]>(
                builder.mesh.positions.len(),
                "Vertex Position",
                builder.devices,
            );

            all_executors.iter().for_each(|(_, _, list)| {
                list.copy_buffer_to_buffer(&pos_vb, &pos_vb_staging);
            });

            (pos_vb, pos_vb_staging)
        };

        let (normal_vb, _normal_vb_staging) = {
            let devices = builder
                .devices
                .iter()
                .filter(|d| builder.normal_vb.contains(d.id))
                .map(|d| *d)
                .collect::<Vec<_>>();

            let executors = all_executors
                .iter()
                .filter(|(d, _, _)| builder.normal_vb.contains(d.id));

            let mut normal_vb_staging = rhi::Buffer::copy::<[f32; 3]>(
                builder.mesh.normals.len(),
                "Vertex Normal",
                &devices,
            );
            normal_vb_staging.write_all(&builder.mesh.normals);

            let normal_vb = rhi::Buffer::vertex::<[f32; 3]>(
                builder.mesh.normals.len(),
                "Vertex Normal",
                &devices,
            );

            executors.for_each(|(_, _, list)| {
                list.copy_buffer_to_buffer(&normal_vb, &normal_vb_staging);
            });

            (normal_vb, normal_vb_staging)
        };

        let (uv_vb, _uv_vb_staging) = {
            let devices = builder
                .devices
                .iter()
                .filter(|d| builder.uv_vb.contains(d.id))
                .map(|d| *d)
                .collect::<Vec<_>>();

            let executors = all_executors
                .iter()
                .filter(|(d, _, _)| builder.uv_vb.contains(d.id));

            let mut uv_vb_staging =
                rhi::Buffer::copy::<[f32; 2]>(builder.mesh.normals.len(), "Vertex UV", &devices);
            uv_vb_staging.write_all(&builder.mesh.uvs);

            let uv_vb =
                rhi::Buffer::vertex::<[f32; 2]>(builder.mesh.uvs.len(), "Vertex UV", &devices);

            executors.for_each(|(_, _, list)| {
                list.copy_buffer_to_buffer(&uv_vb, &uv_vb_staging);
            });

            (uv_vb, uv_vb_staging)
        };

        let (tangent_vb, _tangent_vb_staging) = {
            let devices = builder
                .devices
                .iter()
                .filter(|d| builder.tangent_vb.contains(d.id))
                .map(|d| *d)
                .collect::<Vec<_>>();

            let executors = all_executors
                .iter()
                .filter(|(d, _, _)| builder.tangent_vb.contains(d.id));

            let mut tangent_vb_staging = rhi::Buffer::copy::<[f32; 4]>(
                builder.mesh.tangents.len(),
                "Vertex Tangent",
                &devices,
            );
            tangent_vb_staging.write_all(&builder.mesh.tangents);

            let tangent_vb = rhi::Buffer::vertex::<[f32; 4]>(
                builder.mesh.tangents.len(),
                "Vertex Tangent",
                &devices,
            );

            executors.for_each(|(_, _, list)| {
                list.copy_buffer_to_buffer(&tangent_vb, &tangent_vb_staging);
            });

            (tangent_vb, tangent_vb_staging)
        };

        let (ib, _ib_staging) = {
            let mut ib_staging = rhi::Buffer::copy::<u32>(
                builder.mesh.indices.len(),
                format!("{} Index Buffer", "check"),
                builder.devices,
            );
            ib_staging.write_all(&builder.mesh.indices);

            let ib = rhi::Buffer::index_u32(builder.mesh.indices.len(), "Index", builder.devices);

            all_executors.iter().for_each(|(_, _, list)| {
                list.copy_buffer_to_buffer(&ib, &ib_staging);
            });

            (ib, ib_staging)
        };

        let materials = {
            let devices = builder
                .devices
                .iter()
                .filter(|d| builder.materials.contains(d.id))
                .map(|d| *d)
                .collect::<Vec<_>>();

            let mut materials = rhi::Buffer::constant::<GpuMaterial>(
                builder.mesh.materials.len(),
                "Materials Buffer",
                &devices,
            );

            {
                let data = builder
                    .mesh
                    .materials
                    .iter()
                    .map(|m| GpuMaterial {
                        diffuse: m.diffuse_color,
                        fresnel_r0: m.fresnel_r0,
                        roughness: m.roughness,
                    })
                    .collect::<Vec<_>>();

                materials.write_all(&data);
            }

            materials
        };

        let transform = rhi::Buffer::constant::<GpuTransform>(
            FRAMES_IN_FLIGHT,
            "Transform Buffer",
            builder.devices,
        );

        let images = builder
            .mesh
            .images
            .into_iter()
            .map(|img| match img {
                ImageSource::Path(path) => {
                    let image = image::open(&path).expect("Failed to load png").to_rgba8();

                    let texture = rhi::Texture::new(
                        image.width(),
                        image.height(),
                        1,
                        dx::Format::Rgba8Unorm,
                        1,
                        dx::ResourceFlags::empty(),
                        dx::ResourceStates::CopyDest,
                        None,
                        "Texture",
                        builder.devices,
                    );

                    let total_size = texture
                        .get_texture(builder.devices[0].id)
                        .expect("Failed to get texture")
                        .get_size(builder.devices[0], None);

                    let staging =
                        rhi::Buffer::copy::<u8>(total_size, "Staging Buffer", builder.devices);

                    all_executors.iter().for_each(|(_, _, cmd)| {
                        cmd.load_texture_from_memory(&texture, &staging, image.as_raw());
                        cmd.set_texture_barrier(
                            &texture,
                            dx::ResourceStates::PixelShaderResource,
                            None,
                        );
                    });

                    (texture, staging)
                }
                ImageSource::Data(_vec) => {
                    todo!()
                }
            })
            .collect::<Vec<_>>();

        let image_views = images
            .iter()
            .map(|(t, _)| {
                rhi::TextureView::new(
                    builder.devices[0],
                    t.get_texture(builder.devices[0].id)
                        .expect("Failed to get texture"),
                    rhi::TextureViewType::ShaderResource,
                    None,
                )
            })
            .collect();

        let values = all_executors.into_iter().map(|(_, queue, cmd)| {
            queue.push_cmd_buffer(cmd);
            (queue, queue.execute())
        });

        values.for_each(|(d, v)| {
            d.wait_on_cpu(v);
        });

        Self {
            pos_vb,
            normal_vb,
            uv_vb,
            tangent_vb,
            ib,
            gpu_materials: materials,
            transform,
            images: images.into_iter().map(|(i, _)| i).collect(),
            image_views,

            sub_meshes: builder.mesh.sub_meshes,
            materials: builder.mesh.materials,
        }
    }
}
