//! A skeleton, the vertices bound to it, and the clip that moves them.

use glam::{Affine3A, Mat3, Quat, Vec3};
use rtxmw_nif::{
    Block, ControllerKind, FloatKey, Link, NifFile, QuaternionKey, Transform, VectorKey,
};

use crate::mesh::GeometrySpan;
use crate::static_scene::MeshId;

/// How many bones one vertex may follow.
///
/// **Measured, not chosen.** Across the 556 skinned meshes the game ships, 390,381 vertices follow
/// one bone, 72,629 follow two, 20,834 three, 3,602 four and **107 follow five** — two hundredths
/// of one percent. Those hundred lose their smallest share, which is renormalised away across the
/// other four; carrying a fifth slot for them would cost every vertex in the world a fifth of its
/// weight data.
pub const INFLUENCES: usize = 4;

/// Which bones move one vertex, and by how much.
///
/// Weights sum to one. A vertex following fewer than [`INFLUENCES`] bones pads with zero weights,
/// which cost the skinning a multiply-add each and no branch.
#[derive(Debug, Clone, Copy, Default)]
pub struct Influence {
    /// Indices into [`Rig::bones`].
    pub bones: [u8; INFLUENCES],
    pub weights: [f32; INFLUENCES],
}

/// Marks a joint with no parent, which is where a skeleton's walk starts.
pub const NO_PARENT: u16 = u16::MAX;

/// One joint's animation: what the file says it does, channel by channel.
///
/// A channel with no keys means the joint keeps its rest transform in that respect, which is why
/// they are separate rather than one list of poses — most animated joints in this content rotate
/// and nothing else.
#[derive(Debug, Clone, Default)]
pub struct Channel {
    pub rotations: Vec<QuaternionKey>,
    pub translations: Vec<VectorKey>,
    pub scales: Vec<FloatKey>,
}

impl Channel {
    /// Whether this joint moves at all.
    pub fn is_empty(&self) -> bool {
        self.rotations.is_empty() && self.translations.is_empty() && self.scales.is_empty()
    }

    /// The last moment any of the three says anything about.
    fn duration(&self) -> f32 {
        let rotation = self.rotations.last().map_or(0.0, |key| key.time);
        let translation = self.translations.last().map_or(0.0, |key| key.time);
        let scale = self.scales.last().map_or(0.0, |key| key.time);
        rotation.max(translation).max(scale)
    }

    /// Where the joint stands at `time`, over its rest transform where a channel is silent.
    ///
    /// The rest transform is decomposed to supply what the channels do not, which is exact here
    /// because Morrowind scales uniformly — the one case where a scale cannot be recovered from a
    /// matrix is the one the content does not contain.
    ///
    /// **Read linearly between keys whatever the file declared.** Of the 3.4 million rotation keys
    /// the game ships, 97.9% are linear already, and 95.1% of translations; the rest are quadratic
    /// or TCB, whose tangents this drops. See `docs/design.md` §8.92.
    fn pose(&self, time: f32, rest: Affine3A) -> Affine3A {
        let (rest_scale, rest_rotation, rest_translation) = rest.to_scale_rotation_translation();
        let rotation = sample(&self.rotations, time).map_or(rest_rotation, |value| {
            // Stored `w` first, which is not the order `glam` takes.
            Quat::from_xyzw(value[1], value[2], value[3], value[0]).normalize()
        });
        let translation =
            sample(&self.translations, time).map_or(rest_translation, Vec3::from_array);
        let scale = sample(&self.scales, time).map_or(rest_scale, Vec3::splat);
        Affine3A::from_scale_rotation_translation(scale, rotation, translation)
    }
}

/// A skeleton, the clip that poses it, and the vertices that follow it.
///
/// **One rig per mesh**, because the two are bound to each other: the influences are indices into
/// this bone list and are parallel to that mesh's vertices.
#[derive(Debug, Clone)]
pub struct Rig {
    /// The bind-pose mesh this poses.
    pub mesh: MeshId,
    /// One per joint: its parent, or [`NO_PARENT`]. Ordered parents-first, so one forward pass
    /// composes the whole skeleton.
    pub parents: Vec<u16>,
    /// Each joint in its parent's space, where its channel says nothing.
    pub rest: Vec<Affine3A>,
    /// One per joint, empty where the joint does not move.
    pub channels: Vec<Channel>,
    /// One per skinned bone: which joint it is, and what maps the mesh into its space.
    pub bones: Vec<Bone>,
    /// One per vertex of `mesh`.
    pub influences: Vec<Influence>,
    /// How long the clip runs before it repeats.
    pub duration: f32,
}

/// One bone a mesh is skinned to.
#[derive(Debug, Clone, Copy)]
pub struct Bone {
    pub joint: u16,
    /// The mesh's own space to this bone's space, as the bind pose left it.
    ///
    /// **The only record of the bind pose there is.** A NIF's node transforms are whatever pose the
    /// file was saved in, which across all 556 skinned meshes is never the one the skin was bound
    /// at — so posing a joint back to `inverse_bind`'s own inverse is what returns the mesh to the
    /// vertices as stored.
    pub inverse_bind: Affine3A,
}

impl Rig {
    /// Builds the rig for a model, or `None` where nothing in it moves.
    ///
    /// `spans` is where each geometry block's vertices landed in `mesh` — see
    /// [`crate::mesh::GeometrySpan`]. Every one of them is bound to something: a skinned block to
    /// the bones its `NiSkinInstance` names, and an unskinned one rigidly to the node it hangs
    /// from, which is what carries a creature's static pieces along with the skeleton.
    pub(crate) fn from_nif(
        nif: &NifFile,
        mesh: MeshId,
        vertices: usize,
        spans: &[GeometrySpan],
    ) -> Option<Self> {
        let skeleton = Skeleton::of(nif);
        if skeleton.channels.iter().all(Channel::is_empty) {
            return None;
        }

        let mut bones: Vec<Bone> = Vec::new();
        let mut shares: Vec<Vec<Share>> = vec![Vec::new(); vertices];
        for span in spans {
            match nif.resolve(span.skin) {
                Some(Block::Skin(skin)) => {
                    let Some(Block::SkinData(data)) = nif.resolve(skin.data) else {
                        continue;
                    };
                    // The vertices were baked by the walk, so the bind has to undo that before it
                    // maps them into the bone's space.
                    let unbake = span.placement.inverse();
                    let first = bones.len();
                    for (bone, node) in data.bones.iter().zip(&skin.bones) {
                        let joint = skeleton.joint_of(*node)?;
                        bones.push(Bone {
                            joint,
                            inverse_bind: affine_of(&bone.inverse_bind) * unbake,
                        });
                    }
                    for (offset, bone) in data.bones.iter().enumerate() {
                        let index = (first + offset) as u32;
                        for weight in
                            &data.weights[bone.weights.start as usize..bone.weights.end as usize]
                        {
                            let vertex = span.first_vertex as usize + weight.vertex as usize;
                            shares[vertex].push(Share {
                                bone: index,
                                weight: weight.weight,
                            });
                        }
                    }
                }
                // **Rigid rather than unbound.** A block with no skin still moves with the node it
                // hangs from, and binding it there with that node's own rest transform undone is
                // the same arithmetic as a skinned one with a single full-weight bone.
                _ => {
                    let joint = skeleton.joint_of(span.block)?;
                    let index = bones.len() as u32;
                    bones.push(Bone {
                        joint,
                        inverse_bind: skeleton.world[joint as usize].inverse(),
                    });
                    let span_range = span.first_vertex as usize
                        ..(span.first_vertex + span.vertex_count) as usize;
                    for vertex in span_range {
                        shares[vertex].push(Share {
                            bone: index,
                            weight: 1.0,
                        });
                    }
                }
            }
        }
        if bones.is_empty() || bones.len() > usize::from(u8::MAX) {
            return None;
        }

        let duration = skeleton
            .channels
            .iter()
            .map(Channel::duration)
            .fold(0.0f32, f32::max);
        Some(Self {
            mesh,
            parents: skeleton.parents,
            rest: skeleton.rest,
            channels: skeleton.channels,
            bones,
            influences: shares.iter().map(|share| influence_of(share)).collect(),
            duration,
        })
    }
}

/// One bone's claim on one vertex, before the claims are cut down to [`INFLUENCES`].
#[derive(Debug, Clone, Copy)]
struct Share {
    bone: u32,
    weight: f32,
}

/// The strongest [`INFLUENCES`] claims on a vertex, renormalised to sum to one.
///
/// **Renormalised rather than truncated.** A vertex that loses its fifth share would otherwise be
/// dragged toward the origin by the weight that went missing, which reads as a dent rather than as
/// the rounding it is.
fn influence_of(shares: &[Share]) -> Influence {
    let mut strongest: [Share; INFLUENCES] = [Share {
        bone: 0,
        weight: 0.0,
    }; INFLUENCES];
    for share in shares {
        if let Some(weakest) = strongest
            .iter_mut()
            .min_by(|a, b| a.weight.total_cmp(&b.weight))
            && share.weight > weakest.weight
        {
            *weakest = *share;
        }
    }
    let total: f32 = strongest.iter().map(|share| share.weight).sum();
    let scale = if total > 0.0 { 1.0 / total } else { 0.0 };
    let mut influence = Influence::default();
    for (slot, share) in strongest.iter().enumerate() {
        influence.bones[slot] = share.bone as u8;
        influence.weights[slot] = share.weight * scale;
    }
    influence
}

/// The joints of a model, flattened parents-first out of its node graph.
#[derive(Debug, Default)]
struct Skeleton {
    parents: Vec<u16>,
    rest: Vec<Affine3A>,
    /// Each joint in the model's own space at rest, which is what a rigid binding needs.
    world: Vec<Affine3A>,
    channels: Vec<Channel>,
    /// Which joint each block is, parallel to the blocks of the file.
    by_block: Vec<u16>,
}

impl Skeleton {
    /// Walks `nif` from its roots, in an order that puts every parent before its children.
    fn of(nif: &NifFile) -> Self {
        let mut skeleton = Self {
            by_block: vec![NO_PARENT; nif.blocks().len()],
            ..Self::default()
        };
        // Breadth-first, which is what makes the order parents-first: a child is only ever queued
        // by a parent that has already been given its index.
        let mut queue: std::collections::VecDeque<(Link, u16)> =
            nif.roots().iter().map(|&root| (root, NO_PARENT)).collect();
        while let Some((link, parent)) = queue.pop_front() {
            let Some(index) = link.index() else { continue };
            let Some(Block::Node(node)) = nif.resolve(link) else {
                continue;
            };
            let joint = skeleton.parents.len() as u16;
            let rest = affine_of(&node.av.transform);
            skeleton.parents.push(parent);
            skeleton.rest.push(rest);
            skeleton.world.push(match parent {
                NO_PARENT => rest,
                parent => skeleton.world[parent as usize] * rest,
            });
            skeleton
                .channels
                .push(Channel::of(nif, node.av.net.controller));
            skeleton.by_block[index] = joint;
            for &child in &node.children {
                // A child that is a node becomes a joint of its own when it is reached; one that is
                // geometry never does, so what is recorded for it is the joint it hangs from — see
                // `Skeleton::joint_at`.
                if let Some(child_index) = child.index() {
                    skeleton.by_block[child_index] = joint;
                }
                queue.push_back((child, joint));
            }
        }
        skeleton
    }

    /// Which joint a block is, or for a geometry block, which one it hangs from.
    ///
    /// The format stores no parent link, so the walk records one as it descends: every child gets
    /// its parent's joint written against it, and a child that turns out to be a node overwrites
    /// that with its own when it is reached. What is left for a geometry is where it hangs.
    fn joint_of(&self, link: Link) -> Option<u16> {
        let joint = *self.by_block.get(link.index()?)?;
        (joint != NO_PARENT).then_some(joint)
    }
}

impl Channel {
    /// The keyframe controller on a node's chain, as three channels of keys.
    ///
    /// A node carries a chain of controllers driving different things; this takes the first that
    /// moves it and ignores the rest, which at this version is every node in the shipped content.
    fn of(nif: &NifFile, mut link: Link) -> Self {
        while let Some(Block::Controller(controller)) = nif.resolve(link) {
            if controller.kind == ControllerKind::Keyframe
                && let Some(Block::Keyframe(data)) = nif.resolve(controller.data)
            {
                return Self {
                    rotations: data.rotations.keys.clone(),
                    translations: data.translations.keys.clone(),
                    scales: data.scales.keys.clone(),
                };
            }
            link = controller.next;
        }
        Self::default()
    }
}

/// Where a skeleton stands at one moment, and the scratch it was worked out through.
///
/// **The caller owns it** because a frame poses every placement it holds and must not allocate to
/// do it — the two buffers are cleared and refilled rather than built.
#[derive(Debug, Default)]
pub struct Pose {
    /// Every joint in the mesh's own space. Scratch: the bones below are what a skinning pass reads.
    joints: Vec<Affine3A>,
    /// One per [`Rig::bones`], mesh space to mesh space.
    bones: Vec<Affine3A>,
}

impl Pose {
    /// The matrices a skinning pass wants, one per bone.
    pub fn bones(&self) -> &[Affine3A] {
        &self.bones
    }
}

impl Rig {
    /// Poses the skeleton at `time`, wrapping to the clip's own length.
    pub fn pose(&self, time: f32, into: &mut Pose) {
        let time = if self.duration > 0.0 {
            time.rem_euclid(self.duration)
        } else {
            0.0
        };
        // Joints are ordered parents-first, so a joint's parent is already composed when it is
        // reached and one pass does the whole skeleton.
        into.joints.clear();
        into.joints.reserve_exact(self.parents.len());
        for joint in 0..self.parents.len() {
            let local = self.channels[joint].pose(time, self.rest[joint]);
            into.joints.push(match self.parents[joint] {
                NO_PARENT => local,
                parent => into.joints[parent as usize] * local,
            });
        }
        into.bones.clear();
        into.bones.reserve_exact(self.bones.len());
        for bone in &self.bones {
            into.bones
                .push(into.joints[bone.joint as usize] * bone.inverse_bind);
        }
    }
}

/// The value a channel holds at `time`, or `None` where it holds none.
///
/// Clamped rather than extrapolated at either end: a clip's first key is its pose before it starts
/// and its last is the pose it holds after, which is what a loop wants at the seam.
fn sample<K: Keyframe>(keys: &[K], time: f32) -> Option<K::Value> {
    let first = keys.first()?;
    if time <= first.time() {
        return Some(first.value());
    }
    let last = keys.last().expect("a list with a first has a last");
    if time >= last.time() {
        return Some(last.value());
    }
    // The first key at or after `time`; there is one, because `time` is below the last.
    let after = keys.partition_point(|key| key.time() < time);
    let before = &keys[after - 1];
    let after = &keys[after];
    let span = after.time() - before.time();
    let along = if span > 0.0 {
        (time - before.time()) / span
    } else {
        0.0
    };
    Some(K::between(before, after, along))
}

/// A key a channel can be sampled from.
pub trait Keyframe {
    type Value: Copy;
    fn time(&self) -> f32;
    fn value(&self) -> Self::Value;
    fn between(before: &Self, after: &Self, along: f32) -> Self::Value;
}

impl Keyframe for QuaternionKey {
    type Value = [f32; 4];
    fn time(&self) -> f32 {
        self.time
    }
    fn value(&self) -> [f32; 4] {
        self.value
    }
    fn between(before: &Self, after: &Self, along: f32) -> [f32; 4] {
        let quat = |v: [f32; 4]| Quat::from_xyzw(v[1], v[2], v[3], v[0]);
        // Spherical, not linear: a linear blend of two quaternions shortens the vector between
        // them, which on a wide key spacing shows as a joint slowing down in the middle of a turn.
        let blended = quat(before.value).slerp(quat(after.value), along);
        [blended.w, blended.x, blended.y, blended.z]
    }
}

impl Keyframe for VectorKey {
    type Value = [f32; 3];
    fn time(&self) -> f32 {
        self.time
    }
    fn value(&self) -> [f32; 3] {
        self.value
    }
    fn between(before: &Self, after: &Self, along: f32) -> [f32; 3] {
        Vec3::from_array(before.value)
            .lerp(Vec3::from_array(after.value), along)
            .to_array()
    }
}

impl Keyframe for FloatKey {
    type Value = f32;
    fn time(&self) -> f32 {
        self.time
    }
    fn value(&self) -> f32 {
        self.value
    }
    fn between(before: &Self, after: &Self, along: f32) -> f32 {
        before.value + (after.value - before.value) * along
    }
}

/// A NIF transform as the engine's own, transposed out of the row-major form it is stored in.
///
/// **The single most common way to get a NIF importer subtly wrong**, which is why it is one
/// function rather than a conversion written wherever one is needed.
pub fn affine_of(transform: &Transform) -> Affine3A {
    let r = transform.rotation;
    let rotation = Mat3::from_cols(
        Vec3::new(r[0][0], r[1][0], r[2][0]),
        Vec3::new(r[0][1], r[1][1], r[2][1]),
        Vec3::new(r[0][2], r[1][2], r[2][2]),
    );
    Affine3A::from_mat3_translation(
        rotation * transform.scale,
        Vec3::from_array(transform.translation),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quaternion(time: f32, degrees: f32) -> QuaternionKey {
        let half = degrees.to_radians() / 2.0;
        // Stored `w` first, and about `+Z`.
        QuaternionKey {
            time,
            value: [half.cos(), 0.0, 0.0, half.sin()],
        }
    }

    #[test]
    fn a_channel_is_clamped_at_both_ends_and_read_between_its_keys() {
        let keys = [
            FloatKey {
                time: 1.0,
                value: 10.0,
            },
            FloatKey {
                time: 3.0,
                value: 30.0,
            },
        ];
        // **Clamped rather than extrapolated.** Before the first key the clip holds its opening
        // pose and after the last it holds its closing one, which is what a loop wants at the seam.
        assert_eq!(sample(&keys, 0.0), Some(10.0));
        assert_eq!(sample(&keys, 1.0), Some(10.0));
        assert_eq!(sample(&keys, 9.0), Some(30.0));
        // A quarter of the way from 1 to 3 is 1.5, and a quarter of the way from 10 to 30 is 15.
        assert_eq!(sample(&keys, 1.5), Some(15.0));
        assert_eq!(sample(&keys, 2.0), Some(20.0));
        // And a channel with nothing in it says nothing, rather than saying zero.
        assert_eq!(sample::<FloatKey>(&[], 1.0), None);
    }

    #[test]
    fn a_rotation_is_read_around_the_arc_rather_than_across_it() {
        // Ninety degrees apart, sampled at the middle: on the arc that is forty-five, and across
        // the chord it is forty-five too — but the vector has shortened, so a normalised linear
        // blend lands elsewhere. Ninety and thirty apart is where the two visibly differ.
        let keys = [quaternion(0.0, 0.0), quaternion(1.0, 90.0)];
        let midpoint = sample(&keys, 0.5).expect("two keys sample");
        let angle = 2.0 * midpoint[0].acos().to_degrees();
        assert!(
            (angle - 45.0).abs() < 1.0e-3,
            "the midpoint of a ninety degree turn is forty-five, not {angle}"
        );
        // A third of the way is thirty degrees, which a chord blend would put at 30.36.
        let third = sample(&keys, 1.0 / 3.0).expect("two keys sample");
        let angle = 2.0 * third[0].acos().to_degrees();
        assert!(
            (angle - 30.0).abs() < 1.0e-3,
            "a third of a ninety degree turn is thirty, not {angle}"
        );
    }

    /// A two-joint chain: a root that turns, and a child standing ten units along `+X` from it.
    fn chain() -> Rig {
        Rig {
            mesh: MeshId(0),
            parents: vec![NO_PARENT, 0],
            rest: vec![
                Affine3A::IDENTITY,
                Affine3A::from_translation(Vec3::new(10.0, 0.0, 0.0)),
            ],
            channels: vec![
                Channel {
                    rotations: vec![quaternion(0.0, 0.0), quaternion(2.0, 90.0)],
                    ..Channel::default()
                },
                Channel::default(),
            ],
            bones: vec![
                Bone {
                    joint: 0,
                    inverse_bind: Affine3A::IDENTITY,
                },
                Bone {
                    joint: 1,
                    inverse_bind: Affine3A::from_translation(Vec3::new(-10.0, 0.0, 0.0)),
                },
            ],
            influences: Vec::new(),
            duration: 2.0,
        }
    }

    #[test]
    fn a_joint_carries_its_children_with_it() {
        let rig = chain();
        let mut pose = Pose::default();

        // At rest the bind transforms undo the skeleton exactly, so both bones are the identity and
        // a vertex following either of them stays where it was stored.
        rig.pose(0.0, &mut pose);
        for (index, bone) in pose.bones().iter().enumerate() {
            let moved = bone.transform_point3(Vec3::new(3.0, 4.0, 0.0));
            assert!(
                (moved - Vec3::new(3.0, 4.0, 0.0)).length() < 1.0e-4,
                "bone {index} moved a vertex at rest, to {moved}"
            );
        }

        // Halfway through the clip the root has turned 45 degrees. The child stands ten units
        // along the root's `+X` and carries no rotation of its own, so a vertex at the child's own
        // origin lands where that `+X` turned to: ten at 45 degrees is (7.071, 7.071, 0).
        rig.pose(1.0, &mut pose);
        let child = pose.bones()[1].transform_point3(Vec3::new(10.0, 0.0, 0.0));
        let expected = Vec3::new(10.0 * 0.5f32.sqrt(), 10.0 * 0.5f32.sqrt(), 0.0);
        assert!(
            (child - expected).length() < 1.0e-3,
            "a 45 degree turn should carry the child to {expected}, not {child}"
        );

        // **The clip wraps rather than holding**, so four seconds into a two-second clip is the
        // same pose as the start of it.
        let mut wrapped = Pose::default();
        rig.pose(4.0, &mut wrapped);
        rig.pose(0.0, &mut pose);
        let far = wrapped.bones()[1].transform_point3(Vec3::new(10.0, 0.0, 0.0));
        let near = pose.bones()[1].transform_point3(Vec3::new(10.0, 0.0, 0.0));
        assert!(
            (far - near).length() < 1.0e-3,
            "the clip did not wrap: {far} against {near}"
        );
    }
}
