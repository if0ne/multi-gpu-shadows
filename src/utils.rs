pub fn new_uuid() -> u64 {
    rand::random()
}

use std::{marker::PhantomData, sync::atomic::AtomicUsize};

use glam::Vec4Swizzles;

pub fn align(value: u32, alignment: u32) -> u32 {
    (value + (alignment - 1)) & (!(alignment - 1))
}

pub trait MatrixExt {
    fn right(&self) -> glam::Vec3;
    fn up(&self) -> glam::Vec3;
    fn forward(&self) -> glam::Vec3;
}

impl MatrixExt for glam::Mat4 {
    #[inline]
    fn right(&self) -> glam::Vec3 {
        self.x_axis.xyz()
    }

    #[inline]
    fn up(&self) -> glam::Vec3 {
        self.y_axis.xyz()
    }

    #[inline]
    fn forward(&self) -> glam::Vec3 {
        self.z_axis.xyz()
    }
}
