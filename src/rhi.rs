use std::{
    cell::RefCell,
    rc::Rc,
};

use oxidx::dx::{self, IDescriptorHeap, IDevice, IFence};

pub struct DescriptorHeap {
    pub heap: dx::DescriptorHeap,
    pub ty: dx::DescriptorHeapType,
    pub size: usize,
    pub inc_size: usize,
    pub shader_visible: bool,
    pub descriptors: RefCell<Vec<bool>>,
}

impl DescriptorHeap {
    pub fn new(device: dx::Device, ty: dx::DescriptorHeapType, size: usize) -> Self {
        let descriptors = RefCell::new(vec![false; size]);

        let (shader_visible, flags) =
            if ty == dx::DescriptorHeapType::CbvSrvUav || ty == dx::DescriptorHeapType::Sampler {
                (true, dx::DescriptorHeapFlags::ShaderVisible)
            } else {
                (false, dx::DescriptorHeapFlags::empty())
            };

        let inc_size = device.get_descriptor_handle_increment_size(ty);

        let heap = device
            .create_descriptor_heap(&dx::DescriptorHeapDesc::new(ty, size).with_flags(flags))
            .expect("Failed to create descriptor heap");

        Self {
            heap,
            ty,
            size,
            inc_size,
            shader_visible,
            descriptors,
        }
    }

    pub fn alloc(self: &Rc<Self>) -> Descriptor {
        let Some((index, val)) = self
            .descriptors
            .borrow_mut()
            .iter_mut()
            .enumerate()
            .skip_while(|(i, b)| **b == true)
            .next()
        else {
            panic!("Out of memory in descriptor heap");
        };
        *val = true;

        let cpu = self
            .heap
            .get_cpu_descriptor_handle_for_heap_start()
            .forward(index, self.inc_size);
        let gpu = self
            .heap
            .get_gpu_descriptor_handle_for_heap_start()
            .forward(index, self.inc_size);

        Descriptor {
            heap_index: index,
            cpu,
            gpu,
            parent: Rc::clone(self),
        }
    }
}

pub struct Descriptor {
    pub heap_index: usize,
    pub cpu: dx::CpuDescriptorHandle,
    pub gpu: dx::GpuDescriptorHandle,
    pub parent: Rc<DescriptorHeap>,
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        self.parent.descriptors.borrow_mut()[self.heap_index] = false;
    }
}

pub struct Fence {
    pub fence: dx::Fence,
    pub value: u64,
}

impl Fence {
    pub fn new(device: dx::Device) -> Self {
        let fence = device.create_fence(0, dx::FenceFlags::empty()).expect("Failed to create fence");

        Self {
            fence,
            value: 0,
        }
    }

    pub fn wait(&self, value: u64) {
        if self.completed_value() < value {
            let event = dx::Event::create(false, false).expect("Failed to create event");
            self.fence.set_event_on_completion(value, event).expect("Failed to bind fence to event");
            if event.wait(10_000_000) == 0x00000102 {
                panic!("Device lost")
            }
        }
    }

    pub fn completed_value(&self) -> u64 {
        self.fence.get_completed_value()
    }
}
