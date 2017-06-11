//! # Tree Map
//!
//! An immutable ordered map implemented as a balanced 2-3 tree.
//!
//! This type of map is useful if you want to be able to
//! efficiently iterate over the map in an ordered sequence.
//! Keys need to implement `Ord`, but not `Hash`.
//!
//! Items in the map generally need to implement `Clone`,
//! because the shared structure makes it impossible to
//! determine the lifetime of a reference inside the map.
//! When cloning values would be too expensive,
//! use `TreeMap<Rc<T>>` or `TreeMap<Arc<T>>`.

use std::sync::Arc;
use std::cmp::Ordering;
use std::iter::{Iterator, FromIterator};
use std::collections::HashMap;
use std::hash::Hash;
use list::List;

use self::TreeMapNode::{Leaf, Two, Three};
use self::TreeContext::{TwoLeft, TwoRight, ThreeLeft, ThreeMiddle, ThreeRight};



pub struct TreeMap<K, V>(Arc<TreeMapNode<K, V>>);

impl<K, V> Clone for TreeMap<K, V> {
    fn clone(&self) -> TreeMap<K, V> {
        TreeMap(self.0.clone())
    }
}

pub enum TreeMapNode<K, V> {
    Leaf,
    Two(TreeMap<K, V>, Arc<K>, Arc<V>, TreeMap<K, V>),
    Three(TreeMap<K, V>, Arc<K>, Arc<V>, TreeMap<K, V>, Arc<K>, Arc<V>, TreeMap<K, V>),
}

impl<K, V> TreeMap<K, V> {
    pub fn empty() -> TreeMap<K, V> {
        TreeMap(Arc::new(Leaf))
    }

    pub fn singleton(k: K, v: V) -> TreeMap<K, V> {
        TreeMap::two(TreeMap::empty(), Arc::new(k), Arc::new(v), TreeMap::empty())
    }

    pub fn null(&self) -> bool {
        match *self.0 {
            Leaf => true,
            _ => false,
        }
    }

    pub fn iter(&self) -> TreeMapIter<K, V> {
        TreeMapIter::new(self)
    }

    fn two(left: TreeMap<K, V>, k: Arc<K>, v: Arc<V>, right: TreeMap<K, V>) -> TreeMap<K, V> {
        TreeMap(Arc::new(Two(left, k, v, right)))
    }

    fn three(left: TreeMap<K, V>,
             k1: Arc<K>,
             v1: Arc<V>,
             mid: TreeMap<K, V>,
             k2: Arc<K>,
             v2: Arc<V>,
             right: TreeMap<K, V>)
             -> TreeMap<K, V> {
        TreeMap(Arc::new(Three(left, k1, v1, mid, k2, v2, right)))
    }

    fn all_heights(&self) -> Vec<usize> {
        match *self.0 {
            Leaf => vec![0],
            Two(ref left, _, _, ref right) => {
                left.all_heights()
                    .iter()
                    .chain(right.all_heights().iter())
                    .map(|i| i + 1)
                    .collect()
            }
            Three(ref left, _, _, ref mid, _, _, ref right) => {
                left.all_heights()
                    .iter()
                    .chain(mid.all_heights().iter())
                    .chain(right.all_heights().iter())
                    .map(|i| i + 1)
                    .collect()
            }
        }
    }

    pub fn check_valid(&self) -> bool {
        all_eq(self.all_heights())
    }
}

fn all_eq<A, I>(i: I) -> bool
    where I: IntoIterator<Item = A>,
          A: PartialEq
{
    let mut it = i.into_iter();
    match it.next() {
        None => true,
        Some(ref a) => it.all(|ref b| a == b),
    }
}

impl<K: Clone, V: Clone> TreeMap<K, V> {
    pub fn clone_iter(&self) -> TreeMapCloneIter<K, V> {
        TreeMapCloneIter::new(self)
    }
}

impl<K: Ord, V> TreeMap<K, V> {
    pub fn lookup(&self, k: &K) -> Option<Arc<V>> {
        match *self.0 {
            Leaf => None,
            Two(ref left, ref k1, ref v, ref right) => {
                match k.cmp(k1) {
                    Ordering::Equal => Some(v.clone()),
                    Ordering::Less => left.lookup(k),
                    _ => right.lookup(k),
                }
            }
            Three(ref left, ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => {
                match k.cmp(k1) {
                    Ordering::Equal => Some(v1.clone()),
                    c1 => {
                        match (c1, k.cmp(k2)) {
                            (_, Ordering::Equal) => Some(v2.clone()),
                            (Ordering::Less, _) => left.lookup(k),
                            (_, Ordering::Greater) => right.lookup(k),
                            _ => mid.lookup(k),
                        }
                    }
                }
            }
        }
    }

    pub fn insert(&self, k: K, v: V) -> TreeMap<K, V> {
        down(list![], Arc::new(k), Arc::new(v), self.clone())
    }
}

// Iterator

pub enum IterItem<K, V> {
    Consider(TreeMap<K, V>),
    Yield(Arc<K>, Arc<V>),
}

pub enum IterResult<K, V> {
    Next(Arc<K>, Arc<V>),
    Walk,
    Done,
}

pub struct TreeMapIter<K, V> {
    stack: Vec<IterItem<K, V>>,
}

impl<K, V> TreeMapIter<K, V> {
    fn new(m: &TreeMap<K, V>) -> TreeMapIter<K, V> {
        TreeMapIter { stack: vec![IterItem::Consider(m.clone())] }
    }

    fn step(&mut self) -> IterResult<K, V> {
        match self.stack.pop() {
            None => IterResult::Done,
            Some(IterItem::Consider(m)) => {
                match *m.0 {
                    Leaf => return IterResult::Walk,
                    Two(ref left, ref k, ref v, ref right) => {
                        self.stack.push(IterItem::Consider(right.clone()));
                        self.stack.push(IterItem::Yield(k.clone(), v.clone()));
                        self.stack.push(IterItem::Consider(left.clone()));
                        IterResult::Walk
                    }
                    Three(ref left, ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => {
                        self.stack.push(IterItem::Consider(right.clone()));
                        self.stack.push(IterItem::Yield(k2.clone(), v2.clone()));
                        self.stack.push(IterItem::Consider(mid.clone()));
                        self.stack.push(IterItem::Yield(k1.clone(), v1.clone()));
                        self.stack.push(IterItem::Consider(left.clone()));
                        IterResult::Walk
                    }
                }
            }
            Some(IterItem::Yield(k, v)) => IterResult::Next(k, v),
        }
    }
}

impl<K, V> Iterator for TreeMapIter<K, V> {
    type Item = (Arc<K>, Arc<V>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut action = IterResult::Walk;
        loop {
            match action {
                IterResult::Walk => action = self.step(),
                IterResult::Done => return None,
                IterResult::Next(k, v) => return Some((k, v)),
            }
        }
    }
}

pub struct TreeMapCloneIter<K, V> {
    it: TreeMapIter<K, V>,
}

impl<K: Clone, V: Clone> TreeMapCloneIter<K, V> {
    fn new(m: &TreeMap<K, V>) -> TreeMapCloneIter<K, V> {
        TreeMapCloneIter { it: m.iter() }
    }
}

impl<K: Clone, V: Clone> Iterator for TreeMapCloneIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        match self.it.next() {
            None => None,
            Some((k, v)) => Some(((*k).clone(), (*v).clone())),
        }
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for TreeMap<K, V> {
    fn from_iter<T>(i: T) -> Self
        where T: IntoIterator<Item = (K, V)>
    {
        i.into_iter()
            .fold(TreeMap::empty(), |m, (k, v)| m.insert(k, v))
    }
}

impl<'a, K, V> IntoIterator for &'a TreeMap<K, V> {
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = TreeMapIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V> IntoIterator for TreeMap<K, V> {
    type Item = (Arc<K>, Arc<V>);
    type IntoIter = TreeMapIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Conversions

impl<K: Eq + Hash + Ord, V> From<HashMap<K, V>> for TreeMap<K, V> {
    fn from(m: HashMap<K, V>) -> TreeMap<K, V> {
        m.into_iter()
            .fold(TreeMap::empty(), |n, (k, v)| n.insert(k, v))
    }
}

// Insertion

enum TreeContext<K, V> {
    TwoLeft(Arc<K>, Arc<V>, TreeMap<K, V>),
    TwoRight(TreeMap<K, V>, Arc<K>, Arc<V>),
    ThreeLeft(Arc<K>, Arc<V>, TreeMap<K, V>, Arc<K>, Arc<V>, TreeMap<K, V>),
    ThreeMiddle(TreeMap<K, V>, Arc<K>, Arc<V>, Arc<K>, Arc<V>, TreeMap<K, V>),
    ThreeRight(TreeMap<K, V>, Arc<K>, Arc<V>, TreeMap<K, V>, Arc<K>, Arc<V>),
}

// Delightfully, #[derive(Clone)] doesn't seem to be able to
// produce a working implementation of this.
impl<K, V> Clone for TreeContext<K, V> {
    fn clone(&self) -> TreeContext<K, V> {
        match self {
            &TwoLeft(ref k, ref v, ref right) => TwoLeft(k.clone(), v.clone(), right.clone()),
            &TwoRight(ref left, ref k, ref v) => TwoRight(left.clone(), k.clone(), v.clone()),
            &ThreeLeft(ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => {
                ThreeLeft(k1.clone(),
                          v1.clone(),
                          mid.clone(),
                          k2.clone(),
                          v2.clone(),
                          right.clone())
            }
            &ThreeMiddle(ref left, ref k1, ref v1, ref k2, ref v2, ref right) => {
                ThreeMiddle(left.clone(),
                            k1.clone(),
                            v1.clone(),
                            k2.clone(),
                            v2.clone(),
                            right.clone())
            }
            &ThreeRight(ref left, ref k1, ref v1, ref mid, ref k2, ref v2) => {
                ThreeRight(left.clone(),
                           k1.clone(),
                           v1.clone(),
                           mid.clone(),
                           k2.clone(),
                           v2.clone())
            }
        }
    }
}

#[derive(Clone)]
struct KickUp<K, V>(TreeMap<K, V>, Arc<K>, Arc<V>, TreeMap<K, V>);

fn from_zipper<K, V>(ctx: List<TreeContext<K, V>>, tree: TreeMap<K, V>) -> TreeMap<K, V> {
    match ctx.uncons() {
        None => tree,
        Some((x, xs)) => {
            match x.clone() {
                TwoLeft(k1, v1, right) => from_zipper(xs, TreeMap::two(tree, k1, v1, right)),
                TwoRight(left, k1, v1) => from_zipper(xs, TreeMap::two(left, k1, v1, tree)),
                ThreeLeft(k1, v1, mid, k2, v2, right) => {
                    from_zipper(xs, TreeMap::three(tree, k1, v1, mid, k2, v2, right))
                }
                ThreeMiddle(left, k1, v1, k2, v2, right) => {
                    from_zipper(xs, TreeMap::three(left, k1, v1, tree, k2, v2, right))
                }
                ThreeRight(left, k1, v1, mid, k2, v2) => {
                    from_zipper(xs, TreeMap::three(left, k1, v1, mid, k2, v2, tree))
                }
            }
        }
    }
}

fn down<K: Ord, V>(ctx: List<TreeContext<K, V>>,
                   k: Arc<K>,
                   v: Arc<V>,
                   m: TreeMap<K, V>)
                   -> TreeMap<K, V> {
    match *m.0 {
        Leaf => {
            up(ctx,
               KickUp(TreeMap::empty(), k.clone(), v.clone(), TreeMap::empty()))
        }
        Two(ref left, ref k1, ref v1, ref right) => {
            match k.cmp(k1) {
                Ordering::Equal => {
                    from_zipper(ctx,
                                TreeMap::two(left.clone(), k1.clone(), v1.clone(), right.clone()))
                }
                Ordering::Less => {
                    down(ctx.cons(TwoLeft(k1.clone(), v1.clone(), right.clone())),
                         k,
                         v,
                         left.clone())
                }
                _ => {
                    down(ctx.cons(TwoRight(left.clone(), k1.clone(), v1.clone())),
                         k,
                         v,
                         right.clone())
                }
            }
        }
        Three(ref left, ref k1, ref v1, ref mid, ref k2, ref v2, ref right) => {
            match k.cmp(k1) {
                Ordering::Equal => {
                    from_zipper(ctx,
                                TreeMap::three(left.clone(),
                                               k,
                                               v,
                                               mid.clone(),
                                               k2.clone(),
                                               v2.clone(),
                                               right.clone()))
                }
                c1 => {
                    match (c1, k.cmp(k2)) {
                        (_, Ordering::Equal) => {
                            from_zipper(ctx,
                                        TreeMap::three(left.clone(),
                                                       k1.clone(),
                                                       v1.clone(),
                                                       mid.clone(),
                                                       k,
                                                       v,
                                                       right.clone()))
                        }
                        (Ordering::Less, _) => {
                            down(ctx.cons(ThreeLeft(k1.clone(),
                                                    v1.clone(),
                                                    mid.clone(),
                                                    k2.clone(),
                                                    v2.clone(),
                                                    right.clone())),
                                 k,
                                 v,
                                 left.clone())
                        }
                        (Ordering::Greater, Ordering::Less) => {
                            down(ctx.cons(ThreeMiddle(left.clone(),
                                                      k1.clone(),
                                                      v1.clone(),
                                                      k2.clone(),
                                                      v2.clone(),
                                                      right.clone())),
                                 k,
                                 v,
                                 mid.clone())
                        }
                        _ => {
                            down(ctx.cons(ThreeRight(left.clone(),
                                                     k1.clone(),
                                                     v1.clone(),
                                                     mid.clone(),
                                                     k2.clone(),
                                                     v2.clone())),
                                 k,
                                 v,
                                 right.clone())
                        }
                    }
                }
            }
        }
    }
}

fn up<K, V>(ctx: List<TreeContext<K, V>>, kickup: KickUp<K, V>) -> TreeMap<K, V> {
    match ctx.uncons() {
        None => {
            match kickup {
                KickUp(left, k, v, right) => TreeMap::two(left, k, v, right),
            }
        }
        Some((x, xs)) => {
            match (x, kickup) {
                (&TwoLeft(ref k1, ref v1, ref right), KickUp(ref left, ref k, ref v, ref mid)) => {
                    from_zipper(xs,
                                TreeMap::three(left.clone(),
                                               k.clone(),
                                               v.clone(),
                                               mid.clone(),
                                               k1.clone(),
                                               v1.clone(),
                                               right.clone()))
                }
                (&TwoRight(ref left, ref k1, ref v1), KickUp(ref mid, ref k, ref v, ref right)) => {
                    from_zipper(xs,
                                TreeMap::three(left.clone(),
                                               k1.clone(),
                                               v1.clone(),
                                               mid.clone(),
                                               k.clone(),
                                               v.clone(),
                                               right.clone()))
                }
                (&ThreeLeft(ref k1, ref v1, ref c, ref k2, ref v2, ref d),
                 KickUp(ref a, ref k, ref v, ref b)) => {
                    up(xs,
                       KickUp(TreeMap::two(a.clone(), k.clone(), v.clone(), b.clone()),
                              k1.clone(),
                              v1.clone(),
                              TreeMap::two(c.clone(), k2.clone(), v2.clone(), d.clone())))
                }
                (&ThreeMiddle(ref a, ref k1, ref v1, ref k2, ref v2, ref d),
                 KickUp(ref b, ref k, ref v, ref c)) => {
                    up(xs,
                       KickUp(TreeMap::two(a.clone(), k1.clone(), v1.clone(), b.clone()),
                              k.clone(),
                              v.clone(),
                              TreeMap::two(c.clone(), k2.clone(), v2.clone(), d.clone())))
                }
                (&ThreeRight(ref a, ref k1, ref v1, ref b, ref k2, ref v2),
                 KickUp(ref c, ref k, ref v, ref d)) => {
                    up(xs,
                       KickUp(TreeMap::two(a.clone(), k1.clone(), v1.clone(), b.clone()),
                              k2.clone(),
                              v2.clone(),
                              TreeMap::two(c.clone(), k.clone(), v.clone(), d.clone())))
                }
            }
        }
    }
}

// QuickCheck

#[cfg(any(test, feature = "quickcheck"))]
use quickcheck::{Arbitrary, Gen};

#[cfg(any(test, feature = "quickcheck"))]
impl<K: Ord + Arbitrary + Sync, V: Arbitrary + Sync> Arbitrary for TreeMap<K, V> {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        TreeMap::from_iter(Vec::<(K, V)>::arbitrary(g))
    }
}

// Tests

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn iterates_in_order() {
        let map = TreeMap::singleton(2, 22)
            .insert(1, 11)
            .insert(3, 33)
            .insert(8, 88)
            .insert(9, 99)
            .insert(4, 44)
            .insert(5, 55)
            .insert(6, 66)
            .insert(7, 77);
        let mut it = map.iter();
        assert_eq!(it.next(), Some((Arc::new(1), Arc::new(11))));
        assert_eq!(it.next(), Some((Arc::new(2), Arc::new(22))));
        assert_eq!(it.next(), Some((Arc::new(3), Arc::new(33))));
        assert_eq!(it.next(), Some((Arc::new(4), Arc::new(44))));
        assert_eq!(it.next(), Some((Arc::new(5), Arc::new(55))));
        assert_eq!(it.next(), Some((Arc::new(6), Arc::new(66))));
        assert_eq!(it.next(), Some((Arc::new(7), Arc::new(77))));
        assert_eq!(it.next(), Some((Arc::new(8), Arc::new(88))));
        assert_eq!(it.next(), Some((Arc::new(9), Arc::new(99))));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn into_iter() {
        let map = TreeMap::singleton(2, 22)
            .insert(1, 11)
            .insert(3, 33)
            .insert(8, 88)
            .insert(9, 99)
            .insert(4, 44)
            .insert(5, 55)
            .insert(6, 66)
            .insert(7, 77);
        let mut vec = vec![];
        for (k, v) in map {
            assert_eq!(*k * 11, *v);
            vec.push(*k)
        }
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
