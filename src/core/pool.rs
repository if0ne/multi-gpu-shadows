use std::{collections::VecDeque, marker::PhantomData, sync::Arc};

#[cfg(target_pointer_width = "64")]
type HandleIdx = u32;
#[cfg(target_pointer_width = "64")]
type GenIdx = u32;

#[cfg(not(target_pointer_width = "64"))]
type HandleIdx = usize;
#[cfg(not(target_pointer_width = "64"))]
type GenIdx = usize;

pub struct Pool<T> {
    items: Vec<Option<T>>,
    free_list: VecDeque<usize>,
    generation: Vec<u32>,
}

impl<T> Pool<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            free_list: VecDeque::new(),
            generation: Vec::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, v: T) -> Handle<T> {
        if let Some(free_idx) = self.free_list.pop_front() {
            self.items[free_idx] = Some(v);

            return Handle {
                id: free_idx as HandleIdx,
                gen: self.generation[free_idx] as GenIdx,
                _marker: PhantomData,
            };
        }

        let id = self.items.len();
        self.items.push(Some(v));
        self.generation.push(0);

        Handle {
            id: id as HandleIdx,
            gen: 0,
            _marker: PhantomData,
        }
    }

    pub fn insert_with<C: PoolFetcher<T>>(&mut self, v: T, ctx: &Arc<C>) -> Holder<C, T> {
        let handle = self.insert(v);

        Holder {
            context: Arc::clone(&ctx),
            handle,
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&'_ T> {
        if !self.is_valid_handle(handle) {
            return None;
        }

        self.items[handle.id as usize].as_ref()
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&'_ mut T> {
        if !self.is_valid_handle(handle) {
            return None;
        }

        self.items[handle.id as usize].as_mut()
    }

    pub fn get_pair(
        &mut self,
        first: Handle<T>,
        second: Handle<T>,
    ) -> Option<(&'_ mut T, &'_ mut T)> {
        if first.id > second.id {
            return self.get_pair(first, second);
        }

        if !self.is_valid_handle(first) && first.id == second.id {
            return None;
        }

        let (left, right) = self.items.split_at_mut(second.id as usize);

        let left = left[first.id as usize].as_mut()?;
        let right = right[second.id as usize].as_mut()?;

        Some((left, right))
    }

    pub fn remove(&mut self, handle: Handle<T>) {
        if !self.is_valid_handle(handle) {
            return;
        }

        self.generation[handle.id as usize] += 1;
        self.items[handle.id as usize] = None;
        self.free_list.push_back(handle.id as usize);
    }

    fn is_valid_handle(&self, handle: Handle<T>) -> bool {
        handle.id < self.items.len() as HandleIdx
            && self.generation[handle.id as usize] == handle.gen
    }
}

pub struct Handle<T> {
    id: HandleIdx,
    gen: GenIdx,
    _marker: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            gen: self.gen,
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for Handle<T> {}

pub trait PoolFetcher<T> {
    type Immutable<'a> where Self: 'a;
    type Mutable<'a> where Self: 'a;

    fn get(&self, handle: Handle<T>) -> Option<Self::Immutable<'_>>;
    fn get_mut(&self, handle: Handle<T>) -> Option<Self::Mutable<'_>>;
}

pub struct Holder<C: PoolFetcher<T>, T> {
    context: Arc<C>,
    handle: Handle<T>,
}

impl<C: PoolFetcher<T>, T> Holder<C, T> {
    pub fn get(&self) -> Option<C::Immutable<'_>> {
        self.context.get(self.handle)
    }

    pub fn get_mut(&self) -> Option<C::Mutable<'_>> {
        self.context.get_mut(self.handle)
    }
} 

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use super::{Pool, PoolFetcher};

    pub struct Item;

    pub struct Ctx {
        foo: Mutex<Pool<Item>>,
    }

    impl PoolFetcher<Item> for Ctx {
        type Immutable<'a> = ();
        type Mutable<'a> = ();
    
        fn get(&self, handle: super::Handle<Item>) -> Option<Self::Immutable<'_>> {
            todo!()
        }
    
        fn get_mut(&self, handle: super::Handle<Item>) -> Option<Self::Mutable<'_>> {
            todo!()
        }
    }
    
}
