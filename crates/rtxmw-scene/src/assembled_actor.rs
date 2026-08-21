//! Building a person out of the dozen files they are stored as.

use std::collections::HashSet;

use glam::{Affine3A, Vec3};
use rtxmw_esm::{BodyRecord, PartSlot};
use rtxmw_nif::{Block, NifFile};
use rtxmw_vfs::Vfs;

use crate::material_table::MaterialTable;
use crate::mesh::Mesh;
use crate::rig::{self, Bone, Rig, Share, Skeleton};
use crate::static_scene::MeshId;

/// One piece of a body, and where on the skeleton it belongs.
#[derive(Debug, Clone)]
pub(crate) struct ActorPart {
    /// Path in the virtual file system.
    pub(crate) model: String,
    /// The node it hangs from.
    pub(crate) bone: &'static str,
    /// What the file calls the shapes of this slot, for a file that holds more than one —
    /// see [`PartSlot::shape_name`].
    pub(crate) shape: &'static str,
    /// Whether it is the authored mesh reflected — see [`PartSlot::is_reflected`].
    pub(crate) reflected: bool,
}

impl ActorPart {
    /// The piece `record` puts in `slot`, or `None` where it names no model.
    pub(crate) fn of(record: &BodyRecord, slot: PartSlot) -> Option<Self> {
        let bone = slot.bone()?;
        let shape = slot.shape_name()?;
        if record.model.is_empty() {
            return None;
        }
        Some(Self {
            model: format!("meshes/{}", record.model.replace('\\', "/")),
            bone,
            shape,
            reflected: slot.is_reflected(),
        })
    }
}

/// A body, and the skeleton that moves it.
#[derive(Debug)]
pub(crate) struct AssembledActor {
    pub(crate) mesh: Mesh,
    pub(crate) rig: Rig,
}

impl AssembledActor {
    /// Puts one together from a base skeleton and the parts hung on it.
    ///
    /// **The skeleton's own body is never drawn.** `base_anim.nif` carries a placeholder for every
    /// part — `Tri Head`, `Tri Chest` — and marks them hidden; `base_anim_female.nif` carries the
    /// same and forgot to. Either way what is drawn is the parts, so only the joints are taken from
    /// the base and the geometry in it is left where it is.
    ///
    /// A part that will not load is left out rather than failing the actor: the shipped data names
    /// art that was removed, and a body missing a wrist is better than no body.
    pub(crate) fn assemble(
        vfs: &Vfs,
        skeleton_path: &str,
        parts: &[ActorPart],
        materials: &mut MaterialTable,
    ) -> Option<Self> {
        let bytes = vfs.read(skeleton_path).ok()?;
        let base = NifFile::parse(&bytes).ok()?;
        let skeleton = Skeleton::of(&base);
        if skeleton.is_still() {
            return None;
        }

        let mut mesh = Mesh::default();
        let mut bones: Vec<Bone> = Vec::new();
        let mut shares: Vec<Vec<Share>> = Vec::new();
        // **A skinned file is a whole region of a body, and gives up one slot of itself at a
        // time.** `B_N_Dark Elf_F_Skins.nif` holds both hands *and* the torso, and the `hand`
        // record and the `chest` record both name it — so it is read once per slot asked of it and
        // cut down to that slot on the way in, or a robe that hides the chest would leave the naked
        // torso showing underneath because the hands dragged it back. Both sides come of the one
        // read, the shapes being `Tri Left Hand` and `Tri Right Hand` and the slot's name carrying
        // no side. A rigid file is the opposite: the whole of one arm, added twice and mirrored.
        let mut taken: HashSet<(&str, &str)> = HashSet::new();
        for part in parts {
            if taken.contains(&(part.model.as_str(), part.shape)) {
                continue;
            }
            let Ok(bytes) = vfs.read(&part.model) else {
                continue;
            };
            let Ok(nif) = NifFile::parse(&bytes) else {
                continue;
            };
            let mut spans = Vec::new();
            let mut piece = Mesh::from_nif_tracked(&nif, materials, &mut spans, Some(part.shape));
            if piece.positions.is_empty() {
                continue;
            }
            let base_vertex = mesh.positions.len();
            let attachment = skeleton.joint_named(&part.bone.to_lowercase());
            // **The parts are authored for the right side, and the left is the mirror of it.** One
            // file supplies both arms; what makes one a left arm is a reflection along the bone's
            // own length, applied between the bone and the mesh. OpenMW does the same and by the
            // same test — `components/sceneutil/attach.cpp:166` looks for `Left` in the bone's name.
            let mirrored = part.reflected;
            let attach_bind = match mirrored {
                true => Affine3A::from_scale(Vec3::new(-1.0, 1.0, 1.0)),
                false => Affine3A::IDENTITY,
            };
            let mut bound = Vec::new();
            for span in &spans {
                bound.clear();
                match nif.resolve(span.skin) {
                    // **Armour and clothing span bones**, and carry their own copy of the skeleton
                    // to say which: the names are what tie it back to the one being posed.
                    Some(Block::Skin(skin)) => {
                        let Some(Block::SkinData(data)) = nif.resolve(skin.data) else {
                            continue;
                        };
                        let unbake = span.placement.inverse();
                        for (bone, node) in data.bones.iter().zip(&skin.bones) {
                            let named = match nif.resolve(*node) {
                                Some(Block::Node(node)) => node.av.net.name.to_lowercase(),
                                _ => continue,
                            };
                            let Some(joint) = skeleton.joint_named(&named) else {
                                continue;
                            };
                            bound.push(bones.len() as u32);
                            bones.push(Bone {
                                joint,
                                inverse_bind: rig::affine_of(&bone.inverse_bind) * unbake,
                            });
                        }
                        if bound.len() != data.bones.len() {
                            // A bone this skeleton does not have leaves the weights it owned
                            // unaccounted for, and the part would hang off whatever is left.
                            continue;
                        }
                        for (offset, bone) in data.bones.iter().enumerate() {
                            for weight in &data.weights
                                [bone.weights.start as usize..bone.weights.end as usize]
                            {
                                let vertex = base_vertex
                                    + span.first_vertex as usize
                                    + weight.vertex as usize;
                                grow_to(&mut shares, vertex);
                                shares[vertex].push(Share {
                                    bone: bound[offset],
                                    weight: weight.weight,
                                });
                            }
                        }
                    }
                    // **A plain part is authored in its attachment's own space**, so nothing has to
                    // be undone: the joint's pose is the whole transform.
                    _ => {
                        let Some(joint) = attachment else { continue };
                        let bone = bones.len() as u32;
                        bones.push(Bone {
                            joint,
                            inverse_bind: attach_bind,
                        });
                        for offset in 0..span.vertex_count as usize {
                            let vertex = base_vertex + span.first_vertex as usize + offset;
                            grow_to(&mut shares, vertex);
                            shares[vertex].push(Share { bone, weight: 1.0 });
                        }
                    }
                }
            }
            if spans.iter().any(|span| span.skin.index().is_some()) {
                taken.insert((part.model.as_str(), part.shape));
            }
            // **A reflection turns a triangle inside out**, and the shading normal is chosen by the
            // triangle's own plane — see `Surface::geometric`. Reversing the winding here puts the
            // plane back where the reflection would have left it, so a mirrored arm is lit as its
            // front rather than as its back.
            if mirrored && spans.iter().all(|span| span.skin.is_none()) {
                for triangle in piece.indices.as_chunks_mut::<3>().0 {
                    triangle.swap(1, 2);
                }
            }
            mesh.absorb(&piece);
        }
        if bones.is_empty() || bones.len() > usize::from(u8::MAX) {
            return None;
        }
        grow_to(&mut shares, mesh.positions.len().saturating_sub(1));

        let duration = skeleton.duration();
        let groups = rig::groups_of(&base, duration);
        let playing = rig::resting(&groups).unwrap_or(0.0..duration);
        let influences = shares
            .iter()
            .map(|share| rig::influence_of(share))
            .collect();
        Some(Self {
            rig: skeleton.into_rig(MeshId(0), bones, influences, groups, playing),
            mesh,
        })
    }
}

/// Makes room for a vertex's claims, for a mesh being built one part at a time.
fn grow_to(shares: &mut Vec<Vec<Share>>, vertex: usize) {
    if shares.len() <= vertex {
        shares.resize(vertex + 1, Vec::new());
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::rig::{INFLUENCES, Pose};
    use crate::static_scene::ModelIndex;

    /// How tall a person is in Morrowind's units, which are about 1.4 cm.
    const HEIGHT: f32 = 128.0;

    /// What `B_N_Dark Elf_M_Skins.NIF` holds, counted off its own shapes.
    ///
    /// `Tri Chest` at 284 vertices, and three shapes per hand at 132, 28 and 57 — so 434 for the
    /// pair, and 718 for the whole file.
    const SKINS: &str = "meshes/b/B_N_Dark Elf_M_Skins.NIF";
    const CHEST_VERTICES: usize = 284;
    const HAND_VERTICES: usize = 2 * (132 + 28 + 57);

    /// One part of the skins file, hung where its slot hangs.
    fn skin_part(bone: &'static str, shape: &'static str, reflected: bool) -> ActorPart {
        ActorPart {
            model: SKINS.to_owned(),
            bone,
            shape,
            reflected,
        }
    }

    #[test]
    fn a_skin_file_gives_up_one_slot_at_a_time() {
        let Some(vfs) = rtxmw_vfs::morrowind_archives() else {
            eprintln!("skipping: the game is not installed");
            return;
        };
        // The chest first, then the pair of hands, so a slice of this is one case each.
        let body = [
            skin_part("Chest", "chest", false),
            skin_part("Right Hand", "hand", false),
            skin_part("Left Hand", "hand", true),
        ];

        // **The file is one mesh and three slots point at it**, so what a slot gets has to be cut
        // out by name — otherwise a robe that hides the chest leaves the naked torso showing,
        // because the hands under the sleeves dragged the whole file back in.
        let vertices = |parts: &[ActorPart]| {
            let mut materials = MaterialTable::default();
            AssembledActor::assemble(&vfs, "meshes/base_anim.nif", parts, &mut materials)
                .expect("the skins file should assemble against the base skeleton")
                .mesh
                .positions
                .len()
        };
        assert_eq!(vertices(&body[..1]), CHEST_VERTICES);
        assert_eq!(vertices(&body[1..]), HAND_VERTICES);
        assert_eq!(
            vertices(&body),
            CHEST_VERTICES + HAND_VERTICES,
            "the slots partition the file: nothing shared, nothing left over"
        );

        // **And both hands come of the one read.** The shapes are `Tri Left Hand` and
        // `Tri Right Hand` and the slot's name carries no side, so the left request finds the file
        // already taken for `hand` rather than adding a second pair.
        assert_eq!(vertices(&body[1..2]), HAND_VERTICES);
    }

    #[test]
    fn a_body_assembles_into_a_person_rather_than_a_pile_of_parts() {
        let (Some(vfs), Some(dir)) = (
            rtxmw_vfs::morrowind_archives(),
            rtxmw_vfs::morrowind_data_dir(),
        ) else {
            eprintln!("skipping: the game is not installed");
            return;
        };
        let bytes = std::fs::read(dir.join("Morrowind.esm")).expect("the master should read");
        let esm = rtxmw_esm::EsmReader::new(&bytes).expect("it should parse");
        let index = ModelIndex::build(&esm).expect("the index should build");

        // A guard in Balmora: a male Dark Elf, which is the commonest shape in the game.
        let plan = index
            .actor_of("ordinator")
            .or_else(|| index.actor_of("fargoth"))
            .expect("a named npc should be in the master");
        let mut materials = crate::material_table::MaterialTable::default();
        let actor = AssembledActor::assemble(&vfs, plan.skeleton, &plan.parts, &mut materials)
            .expect("a body should assemble");

        // **Every vertex is claimed exactly once over.** A vertex whose weights fall short is
        // dragged toward the origin and one whose weights overshoot is thrown away from it, and
        // either reads as a body coming apart rather than as the arithmetic it is.
        for (vertex, influence) in actor.rig.influences.iter().enumerate() {
            let total: f32 = influence.weights.iter().sum();
            assert!(
                (total - 1.0).abs() < 1.0e-3,
                "vertex {vertex} of {} carries weights summing to {total}",
                actor.rig.influences.len()
            );
        }
        assert_eq!(
            actor.rig.influences.len(),
            actor.mesh.positions.len(),
            "a rig has one influence per vertex of the mesh it poses"
        );

        // **And posed, it is the size and shape of a person.** The parts are authored in the space
        // of the bone each hangs from, so a body that ignored the skeleton would come out as a
        // dozen pieces stacked at the origin — a metre wide and a hand tall.
        let mut pose = Pose::default();
        actor.rig.pose(0.0, &mut pose);
        let mut lowest = Vec3::splat(f32::INFINITY);
        let mut highest = Vec3::NEG_INFINITY;
        for (vertex, influence) in actor.mesh.positions.iter().zip(&actor.rig.influences) {
            let mut moved = Vec3::ZERO;
            for slot in 0..INFLUENCES {
                let weight = influence.weights[slot];
                if weight == 0.0 {
                    continue;
                }
                moved +=
                    weight * pose.bones()[influence.bones[slot] as usize].transform_point3(*vertex);
            }
            lowest = lowest.min(moved);
            highest = highest.max(moved);
        }
        let size = highest - lowest;
        println!(
            "{}: {} vertices, {} bones, measuring {size}",
            plan.shared_as,
            actor.mesh.positions.len(),
            actor.rig.bones.len()
        );
        assert!(
            (size.z - HEIGHT).abs() < HEIGHT * 0.25,
            "a person is about {HEIGHT} units tall; this one is {:.0}, spanning {lowest} to \
             {highest}",
            size.z
        );
        assert!(
            size.x < HEIGHT * 0.7 && size.y < HEIGHT * 0.7,
            "a person is much taller than they are wide; this one measures {size}"
        );
        // **One arm is the other one reflected.** The parts are authored for the right side and a
        // single file supplies both, so what makes an arm a left arm is a reflection along the
        // bone's own length — see the note at the attachment above. Measured against the
        // skeleton's own placeholder for the part, an upper arm fits its left socket at 3.87 units
        // mirrored against 4.49 as-is, and its right socket the other way round.
        // **And they are dressed.** An `NPC_` carries a list of what it was given and nothing that
        // says which of it is worn, so everything wearable in that list goes on — clothing first
        // and armour over it. A body with nothing but `meshes/b/` in it is one nobody dressed.
        let worn = plan
            .parts
            .iter()
            .filter(|part| !part.model.starts_with("meshes/b/"))
            .count();
        assert!(
            worn > 0,
            "a named townsperson is wearing something; this one has {:?}",
            plan.parts
                .iter()
                .map(|part| &part.model)
                .collect::<Vec<_>>()
        );

        let reflected = actor
            .rig
            .bones
            .iter()
            .filter(|bone| bone.inverse_bind.matrix3.determinant() < 0.0)
            .count();
        let left_attachments = plan.parts.iter().filter(|part| part.reflected).count();
        assert!(
            reflected >= 5,
            "a body has arms and legs; only {reflected} of its bones are reflected"
        );
        assert!(
            reflected <= left_attachments,
            "{reflected} bones are reflected against {left_attachments} left attachments, so \
             something on the right was mirrored too"
        );

        // Standing on the skeleton's own origin, not hanging off it.
        assert!(
            lowest.z.abs() < HEIGHT * 0.15,
            "a body should stand on its own root, not {:.0} units from it",
            lowest.z
        );
    }
}
