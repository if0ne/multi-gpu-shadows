use std::{collections::hash_map::Entry, sync::Arc};

use ahash::HashMap;
use glam::{vec3, Mat4, Quat, Vec3};

use crate::{
    gltf::{GpuMesh, GpuMeshBuilder},
    rhi::{self, FRAMES_IN_FLIGHT},
    GpuTransform,
};

pub struct Scene {
    pub entities: Vec<Entity>,
}

#[derive(Debug)]
pub struct Entity {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: f32,
    pub mesh: MeshHandle,
    pub gpu_transform: rhi::Buffer,
}

impl Entity {
    pub fn new(mesh: MeshHandle, devices: &[&Arc<rhi::Device>]) -> Self {
        let gpu_transform =
            rhi::Buffer::constant::<GpuTransform>(FRAMES_IN_FLIGHT, "Entity Transform", devices);

        Self {
            translation: Default::default(),
            rotation: Default::default(),
            scale: 1.0,
            mesh,
            gpu_transform,
        }
    }

    pub fn update_gpu_resource(&mut self, frame_idx: usize, device: &Arc<rhi::Device>) {
        let matrix = Mat4::from_scale(vec3(self.scale, self.scale, self.scale))
            * Mat4::from_quat(self.rotation)
            * Mat4::from_translation(self.translation);

        self.gpu_transform
            .get_mut_buffer(device.id)
            .expect("Failed to get buffer")
            .write(frame_idx, GpuTransform { world: matrix });
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MeshHandle {
    idx: usize,
    gen: usize,
}

#[derive(Debug, Default)]
pub struct MeshCache {
    next_idx: usize,
    gen: usize,
    name_to_handle: HashMap<String, MeshHandle>,
    handle_to_mesh: HashMap<MeshHandle, GpuMesh>,
}

impl MeshCache {
    pub fn get_mesh_by_name(&mut self, name: impl ToString, builder: GpuMeshBuilder) -> MeshHandle {
        let name = name.to_string();

        match self.name_to_handle.entry(name) {
            Entry::Occupied(occupied_entry) => *occupied_entry.get(),
            Entry::Vacant(vacant_entry) => {
                let gpu_mesh = GpuMesh::new(builder);

                let idx = self.next_idx;
                self.next_idx += 1;

                let handle = MeshHandle { idx, gen: self.gen };

                vacant_entry.insert_entry(handle);
                self.handle_to_mesh.insert(handle, gpu_mesh);

                handle
            }
        }
    }

    pub fn get_mesh(&self, handle: &MeshHandle) -> &GpuMesh {
        self.handle_to_mesh.get(&handle).expect("Cache was cleared")
    }

    pub fn clear(&mut self) {
        self.gen += 1;
        self.next_idx = 0;
        self.name_to_handle.clear();
        self.handle_to_mesh.clear();
    }

    pub fn remove_by_name(&mut self, name: impl AsRef<str>) {
        if let Some(handle) = self.name_to_handle.remove(name.as_ref()) {
            self.handle_to_mesh.remove(&handle);
        }
    }

    pub fn remove_by_handle(&mut self, handle: &MeshHandle) {
        if let Some(shader) = self.handle_to_mesh.remove(handle) {
            // TODO: Remove by name
        }
    }
}
