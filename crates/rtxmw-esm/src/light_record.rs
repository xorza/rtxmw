//! The `LIGH` record's own data, beyond the model every placeable record carries.

use crate::error::Result;
use crate::esm_reader::Record;

/// A light's colour, reach and behaviour, from a `LIGH` record's `LHDT` subrecord.
///
/// Morrowind gives a light a colour and a radius but no intensity — the original renderer had a
/// fixed attenuation curve and no notion of physical units, so brightness came out of the curve
/// rather than the record. Anything reading this has to supply that scale itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightRecord {
    /// Reach in world units.
    pub radius: u32,
    /// Packed `0xAABBGGRR`, as stored.
    pub colour: u32,
    pub flags: u32,
}

impl LightRecord {
    /// Subtracts its light rather than adding it — used to darken, which no ray tracer can do.
    pub const NEGATIVE: u32 = 0x0004;
    /// Carried by the player or an actor, so its position follows them rather than a reference.
    pub const CARRY: u32 = 0x0002;
    /// Starts switched off.
    pub const OFF_BY_DEFAULT: u32 = 0x0020;

    /// The 24 bytes of `LHDT`: weight, value, duration, radius, colour, flags.
    const DATA_SIZE: usize = 24;

    /// Reads the light data, or `None` when the record carries none.
    pub fn parse(record: &Record<'_>) -> Result<Option<Self>> {
        for sub in record.subrecords() {
            let sub = sub?;
            if &sub.name().0 != b"LHDT" {
                continue;
            }
            let data = sub.data();
            if data.len() < Self::DATA_SIZE {
                continue;
            }
            let field = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
            // Weight, value and duration occupy the first twelve bytes and are inventory concerns.
            return Ok(Some(Self {
                radius: field(12),
                colour: field(16),
                flags: field(20),
            }));
        }
        Ok(None)
    }

    /// Whether this light should be placed at all.
    ///
    /// A carried light belongs to whoever holds it rather than to a reference, and a negative one
    /// subtracts illumination — a trick for a renderer that accumulates into a framebuffer, and
    /// meaningless for one that traces paths.
    pub fn is_placeable(&self) -> bool {
        self.flags & (Self::CARRY | Self::NEGATIVE | Self::OFF_BY_DEFAULT) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esm_reader::EsmReader;
    use crate::esm_reader::internals::{SubrecordSpec, push_header, push_record};

    /// A `LIGH` record whose `LHDT` carries the given radius, colour and flags.
    fn light_esm(radius: u32, colour: u32, flags: u32) -> Vec<u8> {
        let mut lhdt = Vec::new();
        // Weight, value and duration come first and are inventory concerns this ignores.
        lhdt.extend_from_slice(&1.5f32.to_le_bytes());
        lhdt.extend_from_slice(&42i32.to_le_bytes());
        lhdt.extend_from_slice(&(-1i32).to_le_bytes());
        lhdt.extend_from_slice(&radius.to_le_bytes());
        lhdt.extend_from_slice(&colour.to_le_bytes());
        lhdt.extend_from_slice(&flags.to_le_bytes());
        assert_eq!(lhdt.len(), LightRecord::DATA_SIZE);

        let mut bytes = Vec::new();
        push_header(&mut bytes);
        push_record(
            &mut bytes,
            b"LIGH",
            0,
            &[
                SubrecordSpec {
                    name: b"NAME",
                    data: b"torch\0",
                },
                SubrecordSpec {
                    name: b"LHDT",
                    data: &lhdt,
                },
            ],
        );
        bytes
    }

    #[test]
    fn the_fields_are_read_from_their_own_offsets() {
        // The three fields the renderer wants sit at bytes 12, 16 and 20 of a 24-byte subrecord,
        // behind three it does not. Values chosen so a four-byte slip in either direction lands on
        // something recognisably wrong rather than a plausible number.
        let bytes = light_esm(0x1234, 0x00AA_BBCC, 0x0004);
        let esm = EsmReader::new(&bytes).expect("should parse");
        let record = esm.records().next().expect("one record").expect("valid");

        let light = LightRecord::parse(&record)
            .expect("valid")
            .expect("the record carries LHDT");
        assert_eq!(light.radius, 0x1234);
        assert_eq!(light.colour, 0x00AA_BBCC);
        assert_eq!(light.flags, 0x0004);
    }

    #[test]
    fn a_record_without_light_data_yields_nothing() {
        // Not every `LIGH` in the wild is complete, and a missing subrecord is data being odd
        // rather than the file being broken.
        let mut bytes = Vec::new();
        push_header(&mut bytes);
        push_record(
            &mut bytes,
            b"LIGH",
            0,
            &[SubrecordSpec {
                name: b"NAME",
                data: b"nameless\0",
            }],
        );
        let esm = EsmReader::new(&bytes).expect("should parse");
        let record = esm.records().next().expect("one record").expect("valid");
        assert!(LightRecord::parse(&record).expect("valid").is_none());
    }

    #[test]
    fn carried_negative_and_disabled_lights_are_not_placed() {
        let plain = LightRecord {
            radius: 256,
            colour: 0x00FF_FFFF,
            flags: 0,
        };
        assert!(plain.is_placeable());
        // Flicker and fire are behaviour, not reasons to skip the light.
        assert!(
            LightRecord {
                flags: 0x0008 | 0x0010,
                ..plain
            }
            .is_placeable()
        );

        for flag in [
            LightRecord::CARRY,
            LightRecord::NEGATIVE,
            LightRecord::OFF_BY_DEFAULT,
        ] {
            assert!(
                !LightRecord {
                    flags: flag,
                    ..plain
                }
                .is_placeable(),
                "{flag:#x}"
            );
        }
    }
}
