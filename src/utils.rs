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

#[derive(Debug)]
pub struct Id<T> {
    pub id: usize,
    pub _marker: PhantomData<T>,
}

impl<T> Id<T> {
    pub fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Self {
            id,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            _marker: self._marker.clone(),
        }
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self._marker == other._marker
    }
}

impl<T> Eq for Id<T> {}
