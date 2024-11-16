use std::rc::Rc;

use gltf::Model;
use oxidx::dx::{self, IDebug, IDevice};

mod gltf;
mod rhi;
mod utils;

fn main() {
    let device = Rc::new(rhi::Device::new(true));

    let queue = rhi::CommandQueue::new(&device, dx::CommandListType::Direct);
    let fence = rhi::Fence::new(&device);

    let gpu_tracker = rhi::GpuResourceTracker::new(&device);

    let model = Model::load(&device, &gpu_tracker, &queue, &fence, "./assets/fantasy_island/scene.gltf");
}
